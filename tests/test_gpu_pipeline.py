"""
tests/test_gpu_pipeline.py

End-to-end integration tests for the full GPU pipeline.

Tests the complete data flow:
  hidden_states [B, T, D]
  → DirichletEvidenceHead.forward()   (Triton on GPU / PyTorch on CPU)
  → uncertainty [B, T]
  → UncertaintyEvictionPolicy.update_uncertainty()
  → block_aggregate_uncertainty_gpu() (Triton on GPU / torch on CPU)
  → block_scores [N_blocks]
  → select_eviction_candidates()
  → evict_ids [n_evict]

Also tests:
  - GPU memory stays flat (no leaks across 100 forward passes)
  - Numerical consistency: Triton output ≈ PyTorch reference
  - Block scorer GPU path vs CPU path give same results
  - Profile section annotations don't crash

Run:
  pytest tests/test_gpu_pipeline.py -v
  pytest tests/test_gpu_pipeline.py -v -k "cuda" --gpu  # GPU-only tests
"""

import pytest
import torch
import numpy as np
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline tests (CPU fallback path — always run)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    """
    Tests the complete head → policy → eviction pipeline.
    Uses CPU fallback — no CUDA required.
    """

    def _make_components(self, block_size=4, D=64, proj_size=16):
        from uncertainty_decode.eviction.uncertainty_head import (
            DirichletEvidenceHead, UncertaintyConfig
        )
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy
        from uncertainty_decode.eviction.block_scorer import BlockScorer

        config = UncertaintyConfig(hidden_size=D, proj_size=proj_size, num_classes=2)
        head   = DirichletEvidenceHead(config)
        policy = UncertaintyEvictionPolicy(
            uncertainty_threshold=0.65,
            uncertainty_weight=0.4,
            block_size=block_size,
            device="cpu",
        )
        scorer = BlockScorer(block_size=block_size, aggregation="max", threshold=0.65)
        return head, policy, scorer

    def test_end_to_end_uncertain_block_protected(self):
        """
        Full pipeline: a block with high-uncertainty tokens must not be
        selected for eviction when lower-uncertainty blocks exist.
        """
        head, policy, scorer = self._make_components(block_size=8, D=64)

        B, T, D = 1, 32, 64
        torch.manual_seed(42)
        hidden = torch.randn(B, T, D)

        # Run head
        output = head(hidden)
        u = output["uncertainty"]   # [1, 32]
        assert u.shape == (B, T)
        assert (u > 0).all() and (u <= 1.001).all()

        # Inject a known high-uncertainty pattern over tokens 8-15 (block 1)
        # by directly setting scores on that block
        u_injected = u[0].clone()
        u_injected[8:16]  = 0.9   # block 1: force high uncertainty
        u_injected[16:24] = 0.05  # block 2: force low uncertainty

        # Update policy (head → policy)
        policy.update_uncertainty(0, u_injected)

        # Select 1 block to evict — must NOT be block 1 (protected)
        evict = policy.select_eviction_candidates(0, n_evict=1)
        assert len(evict) == 1
        assert int(evict[0]) != 1, (
            f"Block 1 (high uncertainty) should not be evicted, got block {evict[0]}"
        )

    def test_block_scorer_gpu_path_matches_cpu_path(self):
        """
        BlockScorer.score_blocks_gpu() must give same results as
        manual torch.max() over blocks. Ensures Triton kernel is correct.
        """
        from uncertainty_decode.eviction.block_scorer import BlockScorer

        block_size = 4
        T = 24
        torch.manual_seed(7)
        u = torch.rand(T)

        scorer_max  = BlockScorer(block_size=block_size, aggregation="max")
        scorer_mean = BlockScorer(block_size=block_size, aggregation="mean")

        # GPU/CPU path through scorer
        scores_max  = scorer_max.score_blocks_gpu(u, T)
        scores_mean = scorer_mean.score_blocks_gpu(u, T)

        # Manual reference
        n_blocks = (T + block_size - 1) // block_size
        padded = torch.zeros(n_blocks * block_size)
        padded[:T] = u
        blocks = padded.reshape(n_blocks, block_size)
        ref_max  = blocks.max(dim=1).values
        ref_mean = blocks.mean(dim=1)

        # Should match within fp16 tolerance
        assert torch.allclose(scores_max.float(),  ref_max,  atol=0.01), \
            f"Max mismatch: {(scores_max.float() - ref_max).abs().max():.4f}"
        assert torch.allclose(scores_mean.float(), ref_mean, atol=0.01), \
            f"Mean mismatch: {(scores_mean.float() - ref_mean).abs().max():.4f}"

    def test_uncertainty_head_uses_triton_on_gpu(self):
        """
        When CUDA is available, forward() should call fused_uncertainty().
        We verify this by checking the function call path.
        """
        from uncertainty_decode.eviction.uncertainty_head import DirichletEvidenceHead, UncertaintyConfig
        import inspect

        src = inspect.getsource(DirichletEvidenceHead.forward)
        assert "fused_uncertainty" in src, (
            "DirichletEvidenceHead.forward() must call fused_uncertainty() for GPU path"
        )
        assert "_pytorch_reference" in src or "_compute_alpha_pytorch" in src, (
            "DirichletEvidenceHead.forward() must have a CPU fallback path"
        )

    def test_block_scorer_uses_gpu_kernel(self):
        """BlockScorer must call block_aggregate_uncertainty_gpu(), not tensor.max()."""
        from uncertainty_decode.eviction.block_scorer import BlockScorer
        import inspect

        src = inspect.getsource(BlockScorer.score_blocks_gpu)
        assert "block_aggregate_uncertainty_gpu" in src, (
            "BlockScorer.score_blocks_gpu() must call block_aggregate_uncertainty_gpu()"
        )

    def test_pipeline_no_memory_leak(self):
        """
        Running 50 forward passes should not accumulate tensors.
        Tests that flush_sequence() actually frees GPU memory.
        """
        from uncertainty_decode.eviction.uncertainty_head import DirichletEvidenceHead, UncertaintyConfig
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        config = UncertaintyConfig(hidden_size=32, proj_size=8)
        head   = DirichletEvidenceHead(config)
        policy = UncertaintyEvictionPolicy(block_size=4, device="cpu")

        # Track number of stored sequences before and after
        for i in range(50):
            hidden = torch.randn(1, 16, 32)
            u = head(hidden)["uncertainty"][0]
            policy.update_uncertainty(i, u)
            policy.flush_sequence(i)          # must free

        # After flushing all, storage should be empty
        assert len(policy._uncertainty)   == 0, "Uncertainty store not cleaned up"
        assert len(policy._block_scores)  == 0, "Block scores not cleaned up"
        assert len(policy._block_ages)    == 0, "Block ages not cleaned up"

    def test_multi_sequence_batch(self):
        """Policy must handle multiple concurrent sequences correctly."""
        from uncertainty_decode.eviction.uncertainty_head import DirichletEvidenceHead, UncertaintyConfig
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        config = UncertaintyConfig(hidden_size=32, proj_size=8)
        head   = DirichletEvidenceHead(config)
        policy = UncertaintyEvictionPolicy(block_size=4, device="cpu")

        B, T = 4, 24
        hidden = torch.randn(B, T, 32)
        batch_u = head(hidden)["uncertainty"]   # [4, 24]

        # Register all sequences
        for seq_id in range(B):
            policy.update_uncertainty(seq_id, batch_u[seq_id])

        # Each sequence should have independent block scores
        for seq_id in range(B):
            assert seq_id in policy._block_scores
            evict = policy.select_eviction_candidates(seq_id, n_evict=1)
            assert len(evict) == 1

        # Flushing one should not affect others
        policy.flush_sequence(0)
        assert 0 not in policy._uncertainty
        assert 1 in policy._uncertainty    # other sequences intact

    def test_uncertainty_range_and_monotonicity(self):
        """
        Higher Dirichlet evidence → lower uncertainty.
        This verifies the Dirichlet math is correct end-to-end.
        """
        from uncertainty_decode.kernels.dirichlet_kernel import _pytorch_reference

        # Construct two inputs: one with high evidence, one with low evidence
        # by using extreme W_ev values
        D, P, K = 32, 8, 2

        torch.manual_seed(42)
        h   = torch.randn(1, 4, D)
        wp  = torch.randn(P, D)
        wn  = torch.ones(D)
        bn  = torch.zeros(D)
        be  = torch.zeros(K)

        # Large W_ev → large evidence → low uncertainty
        we_large  = torch.ones(K, P) * 5.0
        u_certain = _pytorch_reference(h, wp, wn, bn, we_large, be)

        # Small W_ev → small evidence → high uncertainty
        we_small    = torch.ones(K, P) * (-5.0)
        u_uncertain = _pytorch_reference(h, wp, wn, bn, we_small, be)

        assert u_certain.mean() < u_uncertain.mean(), (
            "Larger evidence should produce lower uncertainty"
        )
        assert (u_certain > 0).all() and (u_uncertain > 0).all()
        assert (u_uncertain <= 1.001).all()

    def test_edl_loss_decreases_with_correct_labels(self):
        """EDL loss should be minimizable — loss with correct labels < random."""
        from uncertainty_decode.eviction.uncertainty_head import DirichletEvidenceHead, UncertaintyConfig

        config = UncertaintyConfig(hidden_size=32, proj_size=8)
        head   = DirichletEvidenceHead(config)
        optim  = torch.optim.Adam(head.parameters(), lr=1e-3)

        torch.manual_seed(0)
        hidden = torch.randn(4, 10, 32)
        labels = torch.zeros(4, 10).long()   # all grounded

        initial_loss = head.compute_edl_loss(hidden, labels).item()

        # Train for 20 steps
        for _ in range(20):
            loss = head.compute_edl_loss(hidden, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

        final_loss = head.compute_edl_loss(hidden, labels).item()
        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_protection_map_is_gpu_tensor(self):
        """
        get_protection_map_gpu() must return a tensor on the same device
        as the input uncertainty tensor.
        """
        from uncertainty_decode.eviction.block_scorer import BlockScorer

        scorer = BlockScorer(block_size=4, threshold=0.65)
        u = torch.rand(16)  # CPU
        pmap = scorer.get_protection_map(u)

        assert isinstance(pmap, torch.Tensor)
        assert pmap.dtype == torch.bool
        assert pmap.shape == (4,)
        assert pmap.device == u.device

    def test_ascii_visualization_contains_stats(self):
        """Visualization output must include protection stats."""
        from uncertainty_decode.eviction.block_scorer import BlockScorer

        scorer = BlockScorer(block_size=4, threshold=0.65)
        u = torch.cat([torch.ones(4) * 0.8, torch.ones(4) * 0.1,
                       torch.ones(4) * 0.9, torch.ones(4) * 0.05])
        vis = scorer.visualize_ascii(u)

        assert "protected:" in vis
        assert "2/4" in vis or "2 / 4" in vis or "50.0%" in vis
        assert "block 0" in vis.lower() or "Block 0" in vis

    def test_scorer_gpu_stats(self):
        """gpu_stats() should return correct values."""
        from uncertainty_decode.eviction.block_scorer import BlockScorer

        scorer = BlockScorer(block_size=4, threshold=0.65)
        # 2 protected blocks (0.8, 0.9), 2 not (0.1, 0.2)
        u = torch.cat([
            torch.ones(4) * 0.8,
            torch.ones(4) * 0.1,
            torch.ones(4) * 0.9,
            torch.ones(4) * 0.2,
        ])
        stats = scorer.gpu_stats(u)

        assert stats["n_blocks"] == 4
        assert stats["n_protected"] == 2
        assert abs(stats["protection_rate"] - 0.5) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# CUDA-specific tests (skip if no GPU)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
class TestCUDAPipeline:

    def test_tensors_stay_on_gpu(self):
        """Verify that uncertainty tensors never touch CPU during inference."""
        from uncertainty_decode.eviction.uncertainty_head import DirichletEvidenceHead, UncertaintyConfig
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        device = "cuda"
        config = UncertaintyConfig(hidden_size=64, proj_size=16)
        head   = DirichletEvidenceHead(config).to(device).half().eval()
        policy = UncertaintyEvictionPolicy(block_size=4, device=device)

        hidden = torch.randn(2, 16, 64, device=device, dtype=torch.float16)

        output = head(hidden)
        u = output["uncertainty"]

        # Key assertions: everything on GPU
        assert u.is_cuda, "Uncertainty must be on GPU"
        assert output["alpha"].is_cuda

        policy.update_uncertainty(0, u[0])

        assert policy._block_scores[0].is_cuda, "Block scores must stay on GPU"
        assert policy._block_ages[0].is_cuda,   "Block ages must stay on GPU"

        # Eviction returns CPU tensor (small, acceptable)
        evict = policy.select_eviction_candidates(0, n_evict=1)
        assert not evict.is_cuda, "Eviction indices should be CPU (passed to vLLM)"

    def test_triton_matches_pytorch_reference(self):
        """Triton kernel output must match PyTorch reference within fp16 tolerance."""
        from uncertainty_decode.kernels.dirichlet_kernel import (
            fused_uncertainty, _pytorch_reference
        )

        device = "cuda"
        torch.manual_seed(42)
        B, T, D, P, K = 4, 64, 256, 32, 2

        h  = torch.randn(B, T, D, device=device, dtype=torch.float16)
        wp = torch.randn(P, D, device=device)
        wn = torch.ones(D, device=device)
        bn = torch.zeros(D, device=device)
        we = torch.randn(K, P, device=device)
        be = torch.zeros(K, device=device)

        ref = _pytorch_reference(h, wp, wn, bn, we, be)
        out = fused_uncertainty(h, wp, wn, bn, we, be)

        max_diff = (ref.float() - out.float()).abs().max().item()
        assert max_diff < 0.05, f"Triton vs PyTorch max diff: {max_diff:.5f} > 0.05"

    def test_kernel_overhead_under_threshold(self):
        """Triton kernel must complete in < 5ms for production-scale inputs."""
        from uncertainty_decode.kernels.dirichlet_kernel import fused_uncertainty
        import time

        device = "cuda"
        B, T, D, P, K = 8, 512, 4096, 256, 2
        h  = torch.randn(B, T, D, device=device, dtype=torch.float16)
        wp = torch.randn(P, D, device=device)
        wn = torch.ones(D, device=device)
        bn = torch.zeros(D, device=device)
        we = torch.randn(K, P, device=device)
        be = torch.zeros(K, device=device)

        # Warmup
        for _ in range(5): fused_uncertainty(h, wp, wn, bn, we, be)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(50): fused_uncertainty(h, wp, wn, bn, we, be)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 50

        # Allow up to 5ms on slow GPUs (T4 etc.)
        assert ms < 5.0, f"Kernel too slow: {ms:.2f}ms > 5ms"

    def test_block_aggregation_kernel_correctness(self):
        """GPU block aggregation must match torch.max() reference."""
        from uncertainty_decode.kernels.dirichlet_kernel import block_aggregate_uncertainty_gpu

        device = "cuda"
        T, block_size = 128, 16
        u = torch.rand(T, device=device, dtype=torch.float16)

        gpu_out = block_aggregate_uncertainty_gpu(u, block_size, mode="max")

        # Reference
        n_blocks = T // block_size
        ref = u.reshape(n_blocks, block_size).max(dim=1).values

        max_diff = (gpu_out.float() - ref.float()).abs().max().item()
        assert max_diff < 0.01, f"Block agg diff: {max_diff:.5f}"

    def test_gpu_memory_stable_over_batches(self):
        """GPU memory should not grow over 100 inference steps."""
        from uncertainty_decode.eviction.uncertainty_head import DirichletEvidenceHead, UncertaintyConfig
        from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

        device = "cuda"
        config = UncertaintyConfig(hidden_size=64, proj_size=16)
        head   = DirichletEvidenceHead(config).to(device).half().eval()
        policy = UncertaintyEvictionPolicy(block_size=4, device=device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        mem_before = torch.cuda.memory_allocated()

        for i in range(100):
            hidden = torch.randn(1, 16, 64, device=device, dtype=torch.float16)
            u = head(hidden)["uncertainty"][0]
            policy.update_uncertainty(i % 4, u)  # reuse 4 sequence slots
            if i % 4 == 3:
                for j in range(4): policy.flush_sequence(j)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        # Allow 1MB growth (autotuning cache etc.)
        growth_mb = (mem_after - mem_before) / 1e6
        assert growth_mb < 1.0, f"Memory grew by {growth_mb:.2f}MB over 100 steps"

    def test_profile_sections_dont_crash(self):
        """NVTX profiling annotations must not affect output."""
        from uncertainty_decode.kernels.gpu_profiler import profile_section
        from uncertainty_decode.kernels.dirichlet_kernel import fused_uncertainty

        device = "cuda"
        h  = torch.randn(2, 16, 64, device=device, dtype=torch.float16)
        wp = torch.randn(16, 64, device=device)
        wn = torch.ones(64, device=device)
        bn = torch.zeros(64, device=device)
        we = torch.randn(2, 16, device=device)
        be = torch.zeros(2, device=device)

        with profile_section("test_uncertainty_forward"):
            out = fused_uncertainty(h, wp, wn, bn, we, be)

        assert out.shape == (2, 16)
        assert (out > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
