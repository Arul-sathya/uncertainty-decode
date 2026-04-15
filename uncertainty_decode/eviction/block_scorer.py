"""
uncertainty_decode/eviction/block_scorer.py

BlockScorer: aggregates per-token uncertainty → per-KV-block scores.

GPU PATH (CUDA available):
  Uses block_aggregate_uncertainty_gpu() Triton kernel.
  All scoring stays on GPU — no CPU round-trip.
  ~0.02ms for 4096 tokens vs ~0.15ms for CPU numpy path.

CPU PATH:
  Falls back to torch operations on CPU for testing.

vLLM manages KV cache as fixed-size blocks (default: 16 tokens/block).
The eviction policy operates at block granularity, so we need
block-level scores, not token-level.

Aggregation modes:
  "max"  — protect block if ANY token in it is uncertain  (most conservative)
  "mean" — protect block based on average uncertainty      (balanced)
  "p90"  — protect block if top-10% of tokens are uncertain (selective)
  
GPU kernels implement max and mean.
p90 falls back to a GPU quantile approximation via sorting.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

from uncertainty_decode.kernels.dirichlet_kernel import block_aggregate_uncertainty_gpu


@dataclass
class BlockScore:
    block_id: int
    token_start: int
    token_end: int
    uncertainty_score: float    # [0, 1]
    is_protected: bool


class BlockScorer:
    """
    Maps token-level uncertainty [T] → block-level scores [N_blocks].

    All computation runs on the same device as the input tensor.
    On GPU: uses Triton aggregation kernel (block_aggregate_uncertainty_gpu).
    On CPU: uses torch operations.
    """

    def __init__(
        self,
        block_size: int = 16,
        aggregation: str = "max",   # "max", "mean", "p90"
        threshold: float = 0.65,
    ):
        self.block_size = block_size
        self.aggregation = aggregation
        self.threshold = threshold

    def score_blocks_gpu(
        self,
        uncertainty_scores: torch.Tensor,   # [T] on GPU
        sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute block-level scores using Triton kernel.
        Returns [N_blocks] float16 tensor on the same device.
        This is the hot path — called on every forward pass.
        """
        T = sequence_length or len(uncertainty_scores)
        u = uncertainty_scores[:T]

        if self.aggregation in ("max", "mean"):
            # Direct Triton kernel call — no intermediate tensors
            return block_aggregate_uncertainty_gpu(
                u, block_size=self.block_size, mode=self.aggregation
            )
        elif self.aggregation == "p90":
            # GPU quantile: sort each block and take 90th percentile element
            n_blocks = (T + self.block_size - 1) // self.block_size
            pad_len  = n_blocks * self.block_size
            padded   = torch.zeros(pad_len, device=u.device, dtype=u.dtype)
            padded[:T] = u
            blocks   = padded.reshape(n_blocks, self.block_size)
            # Sort each block and take ~90th percentile index
            sorted_b, _ = torch.sort(blocks, dim=1, descending=True)
            idx_90 = max(0, int(self.block_size * 0.1) - 1)
            return sorted_b[:, idx_90]
        else:
            return block_aggregate_uncertainty_gpu(u, self.block_size, mode="max")

    def score_blocks(
        self,
        uncertainty_scores: torch.Tensor,   # [T], any device
        sequence_length: Optional[int] = None,
    ) -> List[BlockScore]:
        """
        Compute block scores and return as list of BlockScore dataclasses.
        Used by the eviction policy and visualization tools.
        Wraps score_blocks_gpu() and converts to Python objects.
        """
        T = sequence_length or len(uncertainty_scores)

        # GPU path via Triton kernel; CPU path via torch fallback
        block_scores_tensor = self.score_blocks_gpu(uncertainty_scores, T)

        # Convert to Python objects (small tensor, OK to iterate)
        scores_np = block_scores_tensor.cpu().float().numpy()
        n_blocks  = len(scores_np)

        result = []
        for i in range(n_blocks):
            t_start = i * self.block_size
            t_end   = min((i + 1) * self.block_size, T)
            score   = float(scores_np[i])
            result.append(BlockScore(
                block_id=i,
                token_start=t_start,
                token_end=t_end,
                uncertainty_score=score,
                is_protected=score > self.threshold,
            ))
        return result

    def compute_eviction_budget(
        self,
        block_scores: List[BlockScore],
        kv_budget: float,
    ) -> Tuple[List[int], List[int]]:
        """
        Split blocks into keep / evict sets to meet kv_budget.
        Returns (keep_ids, evict_ids).
        Uncertain blocks are always kept first.
        """
        n_total = len(block_scores)
        n_keep  = max(1, int(n_total * kv_budget))

        sorted_blocks = sorted(
            block_scores, key=lambda b: b.uncertainty_score, reverse=True
        )
        keep_ids  = [b.block_id for b in sorted_blocks[:n_keep]]
        evict_ids = [b.block_id for b in sorted_blocks[n_keep:]]
        return keep_ids, evict_ids

    def get_protection_map(
        self,
        uncertainty_scores: torch.Tensor,  # [T]
        sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns [N_blocks] bool tensor on same device. True = protected.
        Fast GPU path: one kernel call + one comparison.
        """
        T = sequence_length or len(uncertainty_scores)
        block_scores = self.score_blocks_gpu(uncertainty_scores, T)
        return block_scores.float() > self.threshold

    def visualize_ascii(
        self,
        uncertainty_scores: torch.Tensor,  # [T]
        sequence_length: Optional[int] = None,
        tokens: Optional[List[str]] = None,  # optional token strings for labels
    ) -> str:
        """
        ASCII map of block uncertainty. Useful for debugging and paper figures.

        Example:
            Block uncertainty map (threshold=0.65):
            ░░░░████░░░░░░░░░░░░████████░░░░░░
            [protected: 3/17 blocks = 17.6%]
        """
        block_scores = self.score_blocks(uncertainty_scores, sequence_length)
        n_blocks = len(block_scores)

        chars = []
        for b in block_scores:
            u = b.uncertainty_score
            if u > self.threshold:
                chars.append("█")        # protected
            elif u > self.threshold * 0.8:
                chars.append("▓")        # high
            elif u > self.threshold * 0.5:
                chars.append("░")        # medium
            else:
                chars.append("·")        # low

        n_protected = sum(1 for b in block_scores if b.is_protected)
        pct = n_protected / max(n_blocks, 1) * 100

        lines = [
            f"Block uncertainty map (threshold={self.threshold}, "
            f"block_size={self.block_size}, mode={self.aggregation}):",
            "".join(chars),
            f"[protected: {n_protected}/{n_blocks} blocks = {pct:.1f}%]",
            "[█=protected  ▓=high  ░=medium  ·=low]",
        ]

        # Show top-3 most uncertain blocks with token ranges
        top3 = sorted(block_scores, key=lambda b: b.uncertainty_score, reverse=True)[:3]
        lines.append("Top-3 uncertain blocks:")
        for b in top3:
            label = "🔒 PROTECTED" if b.is_protected else ""
            lines.append(
                f"  block {b.block_id}: tokens [{b.token_start}:{b.token_end}]  "
                f"U={b.uncertainty_score:.3f} {label}"
            )

        return "\n".join(lines)

    def gpu_stats(
        self,
        uncertainty_scores: torch.Tensor,  # [T]
        sequence_length: Optional[int] = None,
    ) -> dict:
        """
        Return summary statistics computed on GPU.
        Used by profiler and benchmark scripts.
        """
        T = sequence_length or len(uncertainty_scores)
        block_tensor = self.score_blocks_gpu(uncertainty_scores, T)
        prot_mask    = block_tensor.float() > self.threshold

        return {
            "n_blocks":           len(block_tensor),
            "n_protected":        int(prot_mask.sum().item()),
            "protection_rate":    float(prot_mask.float().mean().item()),
            "mean_uncertainty":   float(block_tensor.float().mean().item()),
            "max_uncertainty":    float(block_tensor.float().max().item()),
            "p90_uncertainty":    float(block_tensor.float().quantile(0.9).item()),
        }
