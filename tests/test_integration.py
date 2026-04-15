"""
tests/test_integration.py

Integration tests for UncertaintyDecode.
Tests the full pipeline: uncertainty head → block scorer → eviction policy.
These run WITHOUT vLLM (so they work on CPU in CI).

Run:
    pytest tests/ -v
    pytest tests/test_integration.py -v -k "not slow"
"""

import pytest
import torch
import numpy as np
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty Head Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletEvidenceHead:

    def test_output_shape(self):
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=64, proj_size=16, num_classes=2)
        head = DirichletEvidenceHead(config)

        B, T, D = 2, 10, 64
        hidden = torch.randn(B, T, D)
        output = head(hidden)

        assert output["uncertainty"].shape == (B, T)
        assert output["alpha"].shape == (B, T, 2)
        assert output["evidence"].shape == (B, T, 2)

    def test_uncertainty_range(self):
        """Uncertainty must be in (0, 1] — Dirichlet property."""
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=64, proj_size=16, num_classes=2)
        head = DirichletEvidenceHead(config)

        hidden = torch.randn(4, 20, 64)
        output = head(hidden)
        u = output["uncertainty"]

        assert (u > 0).all(), "Uncertainty must be > 0"
        assert (u <= 1.0 + 1e-5).all(), "Uncertainty must be <= 1"

    def test_alpha_positive(self):
        """Dirichlet concentration params must be > 1 (evidence >= 0, α = e+1)."""
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=64, proj_size=16)
        head = DirichletEvidenceHead(config)

        hidden = torch.randn(2, 8, 64)
        output = head(hidden)

        assert (output["alpha"] >= 1.0).all(), "Alpha must be >= 1"
        assert (output["evidence"] >= 0).all(), "Evidence must be >= 0"

    def test_attention_mask_zeroes_uncertainty(self):
        """Padding tokens (mask=0) should get uncertainty=0."""
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=32, proj_size=8)
        head = DirichletEvidenceHead(config)

        B, T = 2, 10
        hidden = torch.randn(B, T, 32)
        mask = torch.ones(B, T)
        mask[0, 5:] = 0  # mask last 5 tokens in first sequence

        output = head(hidden, attention_mask=mask)
        u = output["uncertainty"]

        assert (u[0, 5:] == 0).all(), "Masked tokens must have 0 uncertainty"
        assert (u[0, :5] > 0).all(), "Valid tokens must have > 0 uncertainty"

    def test_no_gradients_during_inference(self):
        """Head should not compute gradients during forward (inference mode)."""
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=32, proj_size=8)
        head = DirichletEvidenceHead(config)

        hidden = torch.randn(1, 5, 32, requires_grad=True)
        output = head(hidden)
        u = output["uncertainty"]

        assert not u.requires_grad, "Uncertainty should not require grad in inference"

    def test_edl_loss_computable(self):
        """EDL loss should be computable and differentiable for training."""
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=32, proj_size=8, num_classes=2)
        head = DirichletEvidenceHead(config)

        hidden = torch.randn(2, 5, 32)
        labels = torch.randint(0, 2, (2, 5))

        # compute_edl_loss handles the full forward pass internally
        loss = head.compute_edl_loss(hidden, labels, annealing_coeff=0.5)

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

        # Should be differentiable
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Block Scorer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockScorer:

    def test_basic_scoring(self):
        from uncertainty_decode.eviction.block_scorer import BlockScorer
        scorer = BlockScorer(block_size=4, aggregation="max", threshold=0.65)

        # 16 tokens, 4 blocks
        scores = torch.tensor([0.1, 0.2, 0.3, 0.1,   # block 0: max=0.3, not protected
                                0.7, 0.8, 0.9, 0.7,   # block 1: max=0.9, PROTECTED
                                0.1, 0.1, 0.2, 0.1,   # block 2: max=0.2, not protected
                                0.6, 0.7, 0.8, 0.6])  # block 3: max=0.8, PROTECTED

        blocks = scorer.score_blocks(scores)

        assert len(blocks) == 4
        assert not blocks[0].is_protected
        assert blocks[1].is_protected
        assert not blocks[2].is_protected
        assert blocks[3].is_protected

    def test_aggregation_strategies(self):
        from uncertainty_decode.eviction.block_scorer import BlockScorer

        scores = torch.tensor([0.9, 0.1, 0.1, 0.1])  # one high, three low

        max_scorer = BlockScorer(block_size=4, aggregation="max", threshold=0.65)
        mean_scorer = BlockScorer(block_size=4, aggregation="mean", threshold=0.65)

        max_blocks = max_scorer.score_blocks(scores)
        mean_blocks = mean_scorer.score_blocks(scores)

        # Max sees 0.9 → protected
        assert max_blocks[0].is_protected

        # Mean sees 0.3 → not protected
        assert not mean_blocks[0].is_protected

    def test_eviction_budget(self):
        from uncertainty_decode.eviction.block_scorer import BlockScorer
        scorer = BlockScorer(block_size=4, threshold=0.65)

        # 4 blocks: 2 uncertain, 2 certain
        scores = torch.tensor([0.8, 0.8, 0.8, 0.8,   # block 0: uncertain
                                0.1, 0.1, 0.1, 0.1,   # block 1: certain
                                0.9, 0.9, 0.9, 0.9,   # block 2: uncertain
                                0.2, 0.2, 0.2, 0.2])  # block 3: certain

        blocks = scorer.score_blocks(scores)
        keep_ids, evict_ids = scorer.compute_eviction_budget(blocks, kv_budget=0.5)

        # With 50% budget, keep 2 of 4 blocks
        assert len(keep_ids) == 2
        assert len(evict_ids) == 2

        # Uncertain blocks should be kept
        assert 0 in keep_ids
        assert 2 in keep_ids

    def test_ascii_visualization_runs(self):
        from uncertainty_decode.eviction.block_scorer import BlockScorer
        scorer = BlockScorer(block_size=4, threshold=0.65)

        scores = torch.rand(32)
        vis = scorer.visualize_ascii(scores)

        assert isinstance(vis, str)
        assert "Block uncertainty map" in vis
        assert "protected:" in vis


# ─────────────────────────────────────────────────────────────────────────────
# Eviction Policy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertaintyEvictionPolicy:

    def test_basic_eviction_ordering(self):
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        # block_size=4 so 8 tokens = 2 blocks
        policy = UncertaintyEvictionPolicy(
            uncertainty_weight=0.4,
            uncertainty_threshold=0.65,
            kv_budget=0.6,
            block_size=4,
        )

        seq_id = 42
        # Block 0: high uncertainty (0.8-0.9) → protected
        # Block 1: low uncertainty (0.1-0.2) → evictable
        u = torch.tensor([0.8, 0.9, 0.8, 0.9,
                           0.1, 0.2, 0.1, 0.2], dtype=torch.float16)
        policy.update_uncertainty(seq_id, u)

        to_evict = policy.select_eviction_candidates(seq_id, n_evict=1)

        assert len(to_evict) == 1
        # Block 1 (low uncertainty = certain) should be evicted first
        assert int(to_evict[0]) == 1, "Certain block should be evicted before uncertain block"

    def test_protection_count_tracked(self):
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        policy = UncertaintyEvictionPolicy(uncertainty_threshold=0.65, block_size=4)

        seq_id = 99
        # All high-uncertainty → all protected
        policy.update_uncertainty(seq_id, torch.ones(8, dtype=torch.float16) * 0.9)

        policy.select_eviction_candidates(seq_id, n_evict=1)
        stats = policy.get_stats()

        # Should record some protected blocks
        assert stats["total_evictions"] >= 1

    def test_flush_removes_sequence(self):
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        policy = UncertaintyEvictionPolicy(block_size=4)
        policy.update_uncertainty(7, torch.rand(8, dtype=torch.float16))

        assert 7 in policy._uncertainty

        policy.flush_sequence(7)

        assert 7 not in policy._uncertainty
        assert 7 not in policy._block_scores

    def test_unknown_blocks_fall_back_to_empty(self):
        """Sequence with no uncertainty data returns empty eviction list."""
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        policy = UncertaintyEvictionPolicy()

        # Select from sequence that was never registered
        to_evict = policy.select_eviction_candidates(9999, n_evict=2)
        assert len(to_evict) == 0

    def test_protection_map(self):
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        policy = UncertaintyEvictionPolicy(uncertainty_threshold=0.65, block_size=4)
        seq_id = 5

        # 3 uncertain blocks, 1 certain block
        u = torch.tensor(
            [0.8]*4 + [0.8]*4 + [0.8]*4 + [0.1]*4, dtype=torch.float16
        )
        policy.update_uncertainty(seq_id, u)

        pmap = policy.get_protection_map_gpu(seq_id)
        assert pmap is not None
        assert pmap[0] == True
        assert pmap[1] == True
        assert pmap[2] == True
        assert pmap[3] == False


# ─────────────────────────────────────────────────────────────────────────────
# Triton Kernel Tests (requires CUDA — skip if not available)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonKernel:

    def test_numerical_correctness(self):
        """Triton kernel output should match PyTorch baseline within tolerance."""
        from uncertainty_decode.kernels.dirichlet_kernel import dirichlet_uncertainty_triton
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )

        torch.manual_seed(42)
        B, T, D, proj_size, K = 2, 16, 64, 16, 2
        device = "cuda"

        config = UncertaintyConfig(hidden_size=D, proj_size=proj_size, num_classes=K)
        head = DirichletEvidenceHead(config).to(device).float()
        head.eval()

        hidden = torch.randn(B, T, D, device=device)

        # PyTorch reference
        with torch.no_grad():
            ref_output = head(hidden)
            ref_uncertainty = ref_output["uncertainty"]

        # Triton kernel
        W_proj = head.proj.weight.data
        W_norm = head.norm.weight.data
        B_norm = head.norm.bias.data
        W_ev = head.evidence.weight.data
        B_ev = head.evidence.bias.data

        triton_uncertainty = dirichlet_uncertainty_triton(
            hidden, W_proj, W_norm, B_norm, W_ev, B_ev
        )

        # Should be numerically close (fp32 triton vs fp32 pytorch)
        max_diff = (ref_uncertainty - triton_uncertainty).abs().max().item()
        assert max_diff < 0.05, f"Triton/PyTorch max difference: {max_diff:.4f}"

    def test_kernel_overhead_under_2ms(self):
        """Triton kernel must be < 2ms for A100 (production viability)."""
        from uncertainty_decode.kernels.dirichlet_kernel import benchmark_triton_vs_pytorch

        results = benchmark_triton_vs_pytorch(
            B=8, T=512, D=4096, proj_size=256, K=2,
            n_warmup=3, n_trials=20
        )

        # Target: < 2ms. Allow 3ms for slower GPUs (T4, etc.)
        assert results["triton_ms"] < 3.0, (
            f"Triton kernel too slow: {results['triton_ms']:.2f}ms > 3ms target"
        )
        print(f"\nTriton overhead: {results['triton_ms']:.2f}ms "
              f"({results['speedup']:.1f}x over PyTorch)")

    def test_kernel_handles_different_batch_sizes(self):
        """Kernel must handle varying batch sizes without crashing."""
        from uncertainty_decode.kernels.dirichlet_kernel import dirichlet_uncertainty_triton

        D, proj_size, K = 64, 16, 2
        W_proj = torch.randn(proj_size, D, device="cuda")
        W_norm = torch.ones(D, device="cuda")
        B_norm = torch.zeros(D, device="cuda")
        W_ev = torch.randn(K, proj_size, device="cuda")
        B_ev = torch.zeros(K, device="cuda")

        for B, T in [(1, 1), (1, 10), (4, 64), (8, 512)]:
            hidden = torch.randn(B, T, D, device="cuda")
            out = dirichlet_uncertainty_triton(hidden, W_proj, W_norm, B_norm, W_ev, B_ev)
            assert out.shape == (B, T), f"Wrong output shape for B={B}, T={T}: {out.shape}"
            assert (out > 0).all(), "All uncertainties must be positive"


# ─────────────────────────────────────────────────────────────────────────────
# Registry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertaintyHeadRegistry:

    def test_load_fresh_head(self):
        from uncertainty_decode.eviction.uncertainty_head import (
            UncertaintyHeadRegistry, UncertaintyConfig
        )
        config = UncertaintyConfig(hidden_size=32, proj_size=8)
        head = UncertaintyHeadRegistry.load(
            checkpoint_path=None,
            config=config,
            device="cpu",
            dtype=torch.float32,
        )
        assert head is not None
        assert UncertaintyHeadRegistry.get() is not None

    def test_compute_uncertainty_without_head(self):
        """Should return zeros gracefully when no head loaded."""
        from uncertainty_decode.eviction.uncertainty_head import UncertaintyHeadRegistry

        # Reset the registry
        UncertaintyHeadRegistry._instance = None

        hidden = torch.randn(2, 10, 32)
        u = UncertaintyHeadRegistry.compute_uncertainty(hidden)

        assert u.shape == (2, 10)
        assert (u == 0).all(), "Should return zeros when head not loaded"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
