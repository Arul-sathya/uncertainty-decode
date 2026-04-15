"""
uncertainty_decode/serving/llm.py

UncertaintyDecodeLLM — drop-in replacement for vllm.LLM.
Wraps vLLM's engine with:
  1. DirichletEvidenceHead (uncertainty head)
  2. UncertaintyEvictionPolicy (uncertainty-guided KV eviction)
  3. Forward pass hooks to capture hidden states and feed uncertainty scores

Usage:
    from uncertainty_decode import UncertaintyDecodeLLM

    llm = UncertaintyDecodeLLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        uncertainty_threshold=0.65,
        kv_budget=0.6,
        uncertainty_weight=0.4,
        # All standard vLLM args also accepted:
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
    )

    outputs = llm.generate(prompts, sampling_params)
    scores = llm.get_last_uncertainty_scores()  # [T] for analysis
"""

import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass

# vLLM imports (will resolve at runtime if vLLM is installed)
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import ModelRegistry
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[UncertaintyDecode] vLLM not found. Install with: pip install vllm>=0.8.0")

from uncertainty_decode.eviction.uncertainty_head import (
    DirichletEvidenceHead,
    UncertaintyConfig,
    UncertaintyHeadRegistry,
)
from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy
from uncertainty_decode.eviction.block_scorer import BlockScorer


@dataclass
class UncertaintyDecodeConfig:
    """Configuration for UncertaintyDecode-specific parameters."""
    uncertainty_threshold: float = 0.65
    kv_budget: float = 0.6
    uncertainty_weight: float = 0.4
    head_proj_size: int = 256
    head_checkpoint: Optional[str] = None
    log_uncertainty: bool = False        # emit uncertainty stats per request
    uncertainty_dtype: torch.dtype = torch.float16


class UncertaintyDecodeHook:
    """
    Forward hook that captures hidden states from the final transformer layer
    and runs the Dirichlet uncertainty head on them.

    Registers on the LM head (or last transformer layer) of the target model.
    Writes uncertainty scores into the eviction policy's store.
    """

    def __init__(
        self,
        uncertainty_head: DirichletEvidenceHead,
        eviction_policy: UncertaintyEvictionPolicy,
        layer_name: str = "model.norm",   # final layer norm — works for Llama/Mistral
    ):
        self.head = uncertainty_head
        self.policy = eviction_policy
        self.layer_name = layer_name
        self._current_sequence_ids: List[int] = []
        self._last_scores: Optional[torch.Tensor] = None

        # Hook handle (saved for cleanup)
        self._hook_handle = None

    def register(self, model: torch.nn.Module) -> None:
        """Find the target layer by name and register the forward hook."""
        target_layer = None
        for name, module in model.named_modules():
            if name == self.layer_name or name.endswith(self.layer_name):
                target_layer = module
                break

        if target_layer is None:
            # Fallback: hook the last layer norm we can find
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    target_layer = module

        if target_layer is None:
            print(f"[UncertaintyDecode WARNING] Could not find layer '{self.layer_name}'. "
                  f"Uncertainty-guided eviction disabled (falling back to LRU).")
            return

        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        print(f"[UncertaintyDecode] Registered uncertainty hook on: {self.layer_name}")

    def _hook_fn(
        self,
        module: torch.nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        """
        Called after every forward pass through the hooked layer.
        output: hidden states [B, T, D] (after final LayerNorm)
        """
        try:
            hidden_states = output
            if isinstance(output, tuple):
                hidden_states = output[0]

            if hidden_states is None or hidden_states.dim() < 2:
                return

            # Run uncertainty head (fused Triton kernel path if available)
            uncertainty_output = self.head(hidden_states)
            uncertainty = uncertainty_output["uncertainty"]  # [B, T]

            self._last_scores = uncertainty.detach()

            # Write into eviction policy for each sequence in batch
            for i, seq_id in enumerate(self._current_sequence_ids):
                if i < uncertainty.shape[0]:
                    self.policy.update_uncertainty(seq_id, uncertainty[i])

        except Exception as e:
            # Graceful degradation: don't crash inference on hook failure
            print(f"[UncertaintyDecode] Hook error (falling back to LRU): {e}")

    def set_sequence_ids(self, seq_ids: List[int]) -> None:
        """Tell the hook which sequence IDs correspond to the current batch."""
        self._current_sequence_ids = seq_ids

    def get_last_scores(self) -> Optional[torch.Tensor]:
        return self._last_scores

    def cleanup(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()


class UncertaintyDecodeLLM:
    """
    Drop-in replacement for vllm.LLM with uncertainty-guided KV cache eviction.

    All standard vLLM arguments are passed through transparently.
    Additional arguments control the uncertainty eviction behavior.
    """

    def __init__(
        self,
        model: str,
        # UncertaintyDecode-specific args
        uncertainty_threshold: float = 0.65,
        kv_budget: float = 0.6,
        uncertainty_weight: float = 0.4,
        head_checkpoint: Optional[str] = None,
        log_uncertainty: bool = False,
        # Standard vLLM args (passed through)
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        **vllm_kwargs,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM required. Install: pip install vllm>=0.8.0")

        self.ud_config = UncertaintyDecodeConfig(
            uncertainty_threshold=uncertainty_threshold,
            kv_budget=kv_budget,
            uncertainty_weight=uncertainty_weight,
            head_checkpoint=head_checkpoint,
            log_uncertainty=log_uncertainty,
        )

        # Initialize eviction policy
        self.eviction_policy = UncertaintyEvictionPolicy(
            uncertainty_weight=uncertainty_weight,
            uncertainty_threshold=uncertainty_threshold,
            kv_budget=kv_budget,
        )

        # Initialize uncertainty head
        hidden_size = self._infer_hidden_size(model)
        head_config = UncertaintyConfig(
            hidden_size=hidden_size,
            proj_size=256,
            num_classes=2,
            threshold=uncertainty_threshold,
        )
        self.uncertainty_head = UncertaintyHeadRegistry.load(
            checkpoint_path=head_checkpoint,
            config=head_config,
            dtype=torch.float16,
        )

        # Initialize vLLM engine
        print(f"[UncertaintyDecode] Initializing vLLM engine for {model}...")
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            **vllm_kwargs,
        )

        # Register forward hook
        self._hook = UncertaintyDecodeHook(
            uncertainty_head=self.uncertainty_head,
            eviction_policy=self.eviction_policy,
        )
        self._register_hooks()

        print(
            f"[UncertaintyDecode] Ready.\n"
            f"  threshold={uncertainty_threshold}, kv_budget={kv_budget}, "
            f"weight={uncertainty_weight}"
        )

    def generate(
        self,
        prompts: Union[List[str], str],
        sampling_params: Optional[Any] = None,
        **kwargs,
    ):
        """Generate with uncertainty-guided KV eviction. Same interface as vllm.LLM."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

        # Assign sequence IDs for this batch
        seq_ids = list(range(len(prompts)))
        self._hook.set_sequence_ids(seq_ids)

        outputs = self.llm.generate(prompts, sampling_params, **kwargs)

        if self.ud_config.log_uncertainty:
            self._log_uncertainty_stats(prompts, outputs)

        # Cleanup sequence uncertainty stores
        for seq_id in seq_ids:
            self.eviction_policy.flush_sequence(seq_id)

        return outputs

    def get_last_uncertainty_scores(self) -> Optional[torch.Tensor]:
        """
        Returns uncertainty scores from the most recent forward pass.
        Shape: [B, T] where T is the sequence length processed.
        Useful for analysis, visualization, and paper figures.
        """
        return self._hook.get_last_scores()

    def get_eviction_stats(self) -> Dict:
        """Returns eviction statistics (protection rate, total evictions, etc.)"""
        return self.eviction_policy.get_stats()

    def get_uncertainty_distribution(self) -> Dict:
        """Returns summary stats of uncertainty scores (for paper's analysis section)."""
        return self.eviction_policy.get_uncertainty_distribution()

    def _register_hooks(self) -> None:
        """Register forward hooks on the vLLM model."""
        try:
            # Access the underlying model through vLLM's worker
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            self._hook.register(model)
        except AttributeError:
            print(
                "[UncertaintyDecode WARNING] Could not access model internals. "
                "This may happen with some vLLM versions. "
                "Try: pip install vllm==0.8.5"
            )

    def _infer_hidden_size(self, model_name: str) -> int:
        """Infer hidden dimension from model name (avoid loading the model twice)."""
        name_lower = model_name.lower()
        size_map = {
            "llama-3.1-8b": 4096,
            "llama-3.1-70b": 8192,
            "llama-3.2-1b": 2048,
            "llama-3.2-3b": 3072,
            "mistral-7b": 4096,
            "qwen2.5-7b": 3584,
            "gemma-2-9b": 3584,
        }
        for key, dim in size_map.items():
            if key in name_lower:
                return dim
        print(f"[UncertaintyDecode] Unknown model size for {model_name}, assuming D=4096")
        return 4096

    def _log_uncertainty_stats(self, prompts, outputs) -> None:
        dist = self.eviction_policy.get_uncertainty_distribution()
        if dist:
            print(
                f"[UncertaintyDecode] Uncertainty stats: "
                f"mean={dist.get('mean', 0):.3f}, "
                f"p90={dist.get('p90', 0):.3f}, "
                f"pct_protected={dist.get('pct_above_threshold', 0):.1%}"
            )
