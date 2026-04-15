"""
scripts/profile_kernel.py

Complete GPU profiling script for UncertaintyDecode.
Produces all numbers for Section 3.3 of the paper.

Measurements:
  1. Triton vs PyTorch kernel timing across (B, T) configurations
  2. Memory bandwidth utilization per kernel
  3. SM utilization estimate
  4. GPU memory breakdown: KV cache vs uncertainty overhead
  5. Eviction policy latency: GPU-native vs CPU-round-trip
  6. Chrome trace for visualization

Run options:
  python scripts/profile_kernel.py                     # all measurements
  python scripts/profile_kernel.py --quick             # fast subset
  python scripts/profile_kernel.py --nsight            # minimal output for nsight
  nsys profile python scripts/profile_kernel.py --nsight  # nsight Systems

Output:
  results/profile/
    kernel_timing.json       → Table 3 of the paper
    memory_breakdown.json    → Section 4.4
    roofline_analysis.json   → Figure 3
    trace.json               → Chrome trace (chrome://tracing)
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path


def measure_kernel_timing(output_dir: str, quick: bool = False) -> dict:
    """Triton vs PyTorch timing across configurations."""
    from uncertainty_decode.kernels.dirichlet_kernel import benchmark

    print("\n" + "━"*60)
    print("1. KERNEL TIMING: Triton vs PyTorch")
    print("━"*60)

    configs = (
        [(4, 512), (8, 512), (8, 2048)] if quick
        else [(1,128), (4,512), (8,512), (8,2048), (16,512), (16,2048)]
    )

    results = benchmark(
        configs=configs,
        D=4096,
        proj_size=256,
        K=2,
        n_warmup=10 if not quick else 3,
        n_trials=100 if not quick else 20,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/kernel_timing.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def measure_memory_breakdown(output_dir: str) -> dict:
    """
    GPU memory breakdown showing uncertainty head overhead.

    Snapshots:
      baseline    → just the model loaded
      kv_cache    → after allocating KV cache for a batch
      uncertainty → after storing uncertainty tensors
    """
    if not torch.cuda.is_available():
        print("\n[memory] No CUDA GPU — skipping")
        return {}

    from uncertainty_decode.kernels.gpu_profiler import GPUMemoryTracker
    from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

    print("\n" + "━"*60)
    print("2. GPU MEMORY BREAKDOWN")
    print("━"*60)

    tracker = GPUMemoryTracker()
    tracker.reset_peak()

    # Baseline (empty)
    torch.cuda.empty_cache()
    snap0 = tracker.snapshot("baseline_empty")

    # Simulate KV cache allocation for Llama-3.1-8B
    # KV cache: 2 (K+V) * 32 layers * 8 heads * 128 head_dim * 2 bytes * B * T
    B, T = 8, 2048
    n_layers, n_heads, head_dim = 32, 8, 128
    kv_bytes = 2 * n_layers * n_heads * head_dim * 2  # fp16
    kv_cache = torch.zeros(B * T, kv_bytes // 2, device="cuda", dtype=torch.float16)
    snap1 = tracker.snapshot("kv_cache_allocated")

    # Simulate uncertainty tensor overhead
    policy = UncertaintyEvictionPolicy(device="cuda")
    for seq_id in range(B):
        u = torch.rand(T, device="cuda", dtype=torch.float16)
        policy.update_uncertainty(seq_id, u)
    snap2 = tracker.snapshot("after_uncertainty")

    tracker.print_timeline()

    kv_mb   = snap1.allocated_mb - snap0.allocated_mb
    unc_mb  = snap2.allocated_mb - snap1.allocated_mb
    unc_pct = unc_mb / max(kv_mb, 1) * 100

    print(f"\nSummary:")
    print(f"  KV cache:            {kv_mb:.1f} MB")
    print(f"  Uncertainty tensors: {unc_mb:.1f} MB  ({unc_pct:.1f}% overhead)")
    print(f"  Overhead target:     < 5% of KV cache")

    result = {
        "B": B, "T": T,
        "kv_cache_mb": round(kv_mb, 2),
        "uncertainty_overhead_mb": round(unc_mb, 2),
        "overhead_pct": round(unc_pct, 2),
    }
    with open(f"{output_dir}/memory_breakdown.json", "w") as f:
        json.dump(result, f, indent=2)

    # Cleanup
    del kv_cache
    for i in range(B): policy.flush_sequence(i)
    torch.cuda.empty_cache()

    return result


def measure_roofline(output_dir: str, quick: bool = False) -> dict:
    """
    Roofline analysis for each kernel.
    Computes arithmetic intensity and bandwidth utilization.
    """
    if not torch.cuda.is_available():
        print("\n[roofline] No CUDA GPU — skipping")
        return {}

    from uncertainty_decode.kernels.gpu_profiler import SMUtilizationEstimator
    from uncertainty_decode.kernels.dirichlet_kernel import fused_uncertainty, _pytorch_reference

    print("\n" + "━"*60)
    print("3. ROOFLINE ANALYSIS")
    print("━"*60)

    estimator = SMUtilizationEstimator()
    analyses = []

    configs = [(8, 512)] if quick else [(8, 512), (8, 2048), (16, 2048)]
    D, PROJ, K = 4096, 256, 2

    for B, T in configs:
        BT = B * T
        h  = torch.randn(B, T, D, device="cuda", dtype=torch.float16)
        wp = torch.randn(PROJ, D, device="cuda")
        wn = torch.ones(D, device="cuda")
        bn = torch.zeros(D, device="cuda")
        we = torch.randn(K, PROJ, device="cuda")
        be = torch.zeros(K, device="cuda")

        # Warmup
        for _ in range(5):
            fused_uncertainty(h, wp, wn, bn, we, be)
        torch.cuda.synchronize()

        # Time triton kernel
        t0 = time.perf_counter()
        for _ in range(50):
            fused_uncertainty(h, wp, wn, bn, we, be)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 50

        # Compute FLOPs and bytes for roofline
        # FLOPs: LN(5D) + proj(2*D*PROJ) + GELU(20*PROJ) + ev(2*PROJ*K) + misc
        flops = BT * (5*D + 2*D*PROJ + 20*PROJ + 2*PROJ*K + 10*K)
        # Bytes (fused: read hidden+weights once, write uncertainty once)
        bytes_io = (BT*D + PROJ*D + D + D + K*PROJ + K + BT) * 2

        analysis = estimator.analyze_kernel(
            kernel_name=f"fused_uncertainty B={B} T={T}",
            elapsed_ms=ms,
            bytes_accessed=bytes_io,
            flops=flops,
        )
        analyses.append(analysis)

    estimator.print_roofline_summary(analyses)

    with open(f"{output_dir}/roofline_analysis.json", "w") as f:
        json.dump(analyses, f, indent=2)

    return {"analyses": analyses}


def measure_eviction_latency(output_dir: str) -> dict:
    """
    Compare eviction policy latency:
    - GPU-native path (new): uncertainty tensors stay on GPU, torch.topk
    - CPU round-trip path (old): copy scores to CPU, Python sort
    """
    if not torch.cuda.is_available():
        print("\n[eviction] No CUDA GPU — skipping")
        return {}

    from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

    print("\n" + "━"*60)
    print("4. EVICTION POLICY LATENCY: GPU-native vs CPU round-trip")
    print("━"*60)

    results = {}

    for T in [512, 2048, 4096]:
        policy = UncertaintyEvictionPolicy(device="cuda", block_size=16)
        seq_id = 0
        u = torch.rand(T, device="cuda", dtype=torch.float16)
        policy.update_uncertainty(seq_id, u)

        n_blocks = (T + 15) // 16
        n_evict  = n_blocks // 4  # evict 25% of blocks

        # GPU-native path (our policy)
        for _ in range(10): policy.select_eviction_candidates(seq_id, n_evict)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            policy.select_eviction_candidates(seq_id, n_evict)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000 / 200

        # CPU round-trip path (what a naive implementation would do)
        block_scores_gpu = policy._block_scores[seq_id]
        t0 = time.perf_counter()
        for _ in range(200):
            # Simulate old path: copy to CPU, sort in Python, send back
            scores_cpu = block_scores_gpu.cpu().float().numpy()
            sorted_idx = scores_cpu.argsort()[:n_evict]
            _ = torch.tensor(sorted_idx)
        cpu_rt_ms = (time.perf_counter() - t0) * 1000 / 200

        speedup = cpu_rt_ms / gpu_ms
        print(f"  T={T:5d}: GPU={gpu_ms:.3f}ms  CPU-roundtrip={cpu_rt_ms:.3f}ms  "
              f"Speedup={speedup:.1f}x")

        results[f"T={T}"] = {
            "gpu_ms": round(gpu_ms, 4),
            "cpu_roundtrip_ms": round(cpu_rt_ms, 4),
            "speedup": round(speedup, 2),
        }
        policy.flush_sequence(seq_id)

    with open(f"{output_dir}/eviction_latency.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def export_chrome_trace(output_dir: str):
    """Export profiler trace for chrome://tracing visualization."""
    if not torch.cuda.is_available():
        return

    from uncertainty_decode.kernels.gpu_profiler import UncertaintyProfiler, profile_section
    from uncertainty_decode.kernels.dirichlet_kernel import fused_uncertainty

    print("\n" + "━"*60)
    print("5. CHROME TRACE EXPORT")
    print("━"*60)

    B, T, D, PROJ, K = 8, 512, 4096, 256, 2
    h  = torch.randn(B, T, D, device="cuda", dtype=torch.float16)
    wp = torch.randn(PROJ, D, device="cuda")
    wn = torch.ones(D, device="cuda")
    bn = torch.zeros(D, device="cuda")
    we = torch.randn(K, PROJ, device="cuda")
    be = torch.zeros(K, device="cuda")

    profiler = UncertaintyProfiler()
    profiler.start()

    for _ in range(5):  # 5 iterations = 5 steps in the trace
        with profile_section("uncertainty_head_forward"):
            with profile_section("fused_layernorm_proj"):
                out = fused_uncertainty(h, wp, wn, bn, we, be)

    profiler.stop()
    profiler.print_summary(top_k=10)

    trace_path = f"{output_dir}/trace.json"
    profiler.export_chrome_trace(trace_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/profile")
    parser.add_argument("--quick", action="store_true",
                        help="Fast run for CI / sanity check")
    parser.add_argument("--nsight", action="store_true",
                        help="Minimal output mode (used with nsys/ncu)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    measure_kernel_timing(args.output, quick=args.quick)
    measure_memory_breakdown(args.output)
    measure_roofline(args.output, quick=args.quick)
    measure_eviction_latency(args.output)

    if not args.nsight:
        export_chrome_trace(args.output)

    print(f"\n✓ All profiling complete. Results in {args.output}/")
    print("  Paper numbers:")
    print("  → kernel_timing.json    Section 3.3, Table 3")
    print("  → roofline_analysis.json  Figure 3")
    print("  → eviction_latency.json   Section 3.4")
    print("  → memory_breakdown.json   Section 4.4")
    print("  → trace.json              chrome://tracing")


if __name__ == "__main__":
    main()
