"""
UncertaintyDecode: Hallucination-Aware KV Cache Eviction for LLM Inference

Quick start:
    from uncertainty_decode import UncertaintyDecodeLLM

    llm = UncertaintyDecodeLLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        uncertainty_threshold=0.65,
        kv_budget=0.6,
    )
    outputs = llm.generate(["Your prompt here"])
    uncertainty = llm.get_last_uncertainty_scores()
"""

from uncertainty_decode.serving.llm import UncertaintyDecodeLLM, UncertaintyDecodeConfig
from uncertainty_decode.eviction.uncertainty_head import (
    DirichletEvidenceHead,
    UncertaintyConfig,
    UncertaintyHeadRegistry,
)
from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy
from uncertainty_decode.eviction.block_scorer import BlockScorer

__version__ = "0.1.0"
__author__ = "Arul Sathya Rajasrinivasan"

__all__ = [
    "UncertaintyDecodeLLM",
    "UncertaintyDecodeConfig",
    "DirichletEvidenceHead",
    "UncertaintyConfig",
    "UncertaintyHeadRegistry",
    "UncertaintyEvictionPolicy",
    "BlockScorer",
]
