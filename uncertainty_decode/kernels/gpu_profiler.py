"""
uncertainty_decode/kernels/gpu_profiler.py

GPU profiling utilities for UncertaintyDecode.

WHAT THIS MEASURES
-------------------
1. SM (Streaming Multiprocessor) utilization
   → Are we keeping the GPU busy? Low SM util = launch-overhead bound.

2. Memory bandwidth utilization
   → Are we memory-bound? bytes_read+written / time vs peak BW.

3. Kernel timeline
   → Which kernels dominate? Is the uncertainty head visible overhead?

4. GPU memory tracking
   → KV cache usage, uncertainty tensor overhead, total allocation.

HOW TO USE WITH NSIGHT
-----------------------
Option A — nsight Systems (timeline):
    nsys profile --stats=true python scripts/profile_kernel.py
    nsys-ui report1.nsys-rep

Option B — nsight Compute (kernel metrics):
    ncu --set full -o profile python scripts/profile_kernel.py
    ncu-ui profile.ncu-rep

Option C — torch.profiler (works without nsight):
    profiler = UncertaintyProfiler()
    profiler.start()
    ... run inference ...
    profiler.stop()
    profiler.print_summary()
    profiler.export_chrome_trace("trace.json")  # open in chrome://tracing
"""

import torch
import time
import json
import contextlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from torch.profiler import (
        profile, record_function, ProfilerActivity,
        tensorboard_trace_handler, schedule,
    )
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory tracker — wraps torch.cuda.memory_*
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    label: str
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    timestamp: float = field(default_factory=time.perf_counter)

    def delta(self, other: "MemorySnapshot") -> Dict:
        return {
            "label": self.label,
            "delta_allocated_mb": round(self.allocated_mb - other.allocated_mb, 2),
            "delta_reserved_mb":  round(self.reserved_mb  - other.reserved_mb,  2),
            "peak_mb": round(self.peak_allocated_mb, 2),
        }


class GPUMemoryTracker:
    """
    Track GPU memory usage across the inference pipeline.
    Identifies how much memory the uncertainty tensors add
    on top of the base KV cache.
    """

    def __init__(self):
        self._snapshots: List[MemorySnapshot] = []
        self._enabled = torch.cuda.is_available()

    def snapshot(self, label: str) -> MemorySnapshot:
        if not self._enabled:
            return MemorySnapshot(label, 0, 0, 0)
        snap = MemorySnapshot(
            label=label,
            allocated_mb=torch.cuda.memory_allocated() / 1e6,
            reserved_mb=torch.cuda.memory_reserved() / 1e6,
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
        )
        self._snapshots.append(snap)
        return snap

    def reset_peak(self):
        if self._enabled:
            torch.cuda.reset_peak_memory_stats()

    def print_timeline(self):
        """Print memory timeline showing delta at each checkpoint."""
        if not self._snapshots:
            print("No snapshots recorded.")
            return

        print("\n" + "="*60)
        print("GPU Memory Timeline")
        print("="*60)
        print(f"{'Label':<30} {'Alloc(MB)':>10} {'Δ(MB)':>8} {'Peak(MB)':>10}")
        print("-"*60)

        prev = self._snapshots[0]
        for snap in self._snapshots:
            delta = snap.allocated_mb - prev.allocated_mb
            sign = "+" if delta >= 0 else ""
            print(f"{snap.label:<30} {snap.allocated_mb:>10.1f} "
                  f"{sign}{delta:>7.1f} {snap.peak_allocated_mb:>10.1f}")
            prev = snap
        print("="*60)

    def get_uncertainty_overhead_mb(self) -> float:
        """
        Compute memory overhead of uncertainty tensors specifically.
        Call snapshot('before_uncertainty') and snapshot('after_uncertainty')
        around the update_uncertainty() call.
        """
        before = next((s for s in self._snapshots if 'before_uncertainty' in s.label), None)
        after  = next((s for s in self._snapshots if 'after_uncertainty'  in s.label), None)
        if before and after:
            return after.allocated_mb - before.allocated_mb
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Kernel-level profiler using torch.profiler
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyProfiler:
    """
    Profiles UncertaintyDecode's kernel execution.

    Wraps torch.profiler.profile to measure:
      - CUDA time per kernel (µs)
      - CPU time per operation
      - Memory bandwidth per kernel
      - Comparison: with vs without uncertainty head

    Output:
      - Chrome trace JSON (open in chrome://tracing)
      - Sorted kernel table (like nsight Compute's summary)
      - Roofline data for bandwidth analysis
    """

    def __init__(self, use_cuda: bool = True, record_shapes: bool = True):
        self._prof = None
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._record_shapes = record_shapes
        self._traces: List[Dict] = []
        self._active = False

    def start(self):
        if not PROFILER_AVAILABLE:
            print("[UncertaintyProfiler] torch.profiler not available")
            return

        activities = [ProfilerActivity.CPU]
        if self._use_cuda:
            activities.append(ProfilerActivity.CUDA)

        self._prof = profile(
            activities=activities,
            record_shapes=self._record_shapes,
            profile_memory=True,
            with_stack=False,
        )
        self._prof.__enter__()
        self._active = True

    def stop(self):
        if self._prof and self._active:
            self._prof.__exit__(None, None, None)
            self._active = False

    def export_chrome_trace(self, path: str = "uncertainty_decode_trace.json"):
        """Export to Chrome trace format — open in chrome://tracing."""
        if self._prof:
            self._prof.export_chrome_trace(path)
            print(f"[UncertaintyProfiler] Chrome trace saved to {path}")
            print(f"  Open in: chrome://tracing → Load → select {path}")

    def print_summary(self, top_k: int = 20):
        """Print sorted kernel table like nsight Compute's kernel summary."""
        if not self._prof:
            print("[UncertaintyProfiler] No profile data.")
            return

        print("\n" + "="*80)
        print("KERNEL SUMMARY (sorted by CUDA time)")
        print("="*80)
        print(f"{'Kernel':<45} {'CUDA(µs)':>10} {'CPU(µs)':>10} {'Calls':>7}")
        print("-"*80)

        events = self._prof.key_averages()
        events_sorted = sorted(events, key=lambda e: e.cuda_time_total, reverse=True)

        total_cuda = sum(e.cuda_time_total for e in events_sorted)

        for event in events_sorted[:top_k]:
            name = event.key[:44]
            cuda_us = event.cuda_time_total / max(event.count, 1)
            cpu_us  = event.cpu_time_total  / max(event.count, 1)
            pct = event.cuda_time_total / max(total_cuda, 1) * 100
            print(f"{name:<45} {cuda_us:>9.1f}µ {cpu_us:>9.1f}µ {event.count:>7d}  ({pct:.1f}%)")

        print("="*80)
        print(f"Total CUDA time: {total_cuda/1000:.2f}ms")

    def measure_kernel_overhead(
        self,
        fn_with_uncertainty,
        fn_without_uncertainty,
        n_warmup: int = 5,
        n_trials: int = 50,
    ) -> Dict:
        """
        Measure overhead of uncertainty head vs baseline.

        Returns dict with:
          baseline_ms, with_uncertainty_ms, overhead_ms, overhead_pct
        """
        if not torch.cuda.is_available():
            return {"error": "No CUDA GPU"}

        # Warmup
        for _ in range(n_warmup):
            fn_without_uncertainty()
            fn_with_uncertainty()
        torch.cuda.synchronize()

        # Baseline
        t0 = time.perf_counter()
        for _ in range(n_trials):
            fn_without_uncertainty()
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - t0) * 1000 / n_trials

        # With uncertainty
        t0 = time.perf_counter()
        for _ in range(n_trials):
            fn_with_uncertainty()
        torch.cuda.synchronize()
        with_ms = (time.perf_counter() - t0) * 1000 / n_trials

        overhead_ms  = with_ms - baseline_ms
        overhead_pct = overhead_ms / max(baseline_ms, 1e-6) * 100

        return {
            "baseline_ms":       round(baseline_ms, 3),
            "with_uncertainty_ms": round(with_ms, 3),
            "overhead_ms":       round(overhead_ms, 3),
            "overhead_pct":      round(overhead_pct, 2),
            "n_trials":          n_trials,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SM utilization estimator
# ─────────────────────────────────────────────────────────────────────────────

class SMUtilizationEstimator:
    """
    Estimate SM utilization from achieved throughput vs peak.

    NVIDIA doesn't expose SM utilization from Python without nvidia-smi/NVML,
    but we can estimate it from:
      achieved_GFLOPS / peak_GFLOPS * 100

    For memory-bound kernels (which ours are), bandwidth utilization
    is more meaningful:
      achieved_BW / peak_BW * 100
    """

    PEAK_SPECS = {
        "A100": {"bw_gb_s": 2000, "fp16_tflops": 312, "sm_count": 108},
        "H100": {"bw_gb_s": 3350, "fp16_tflops": 989, "sm_count": 132},
        "A10":  {"bw_gb_s": 600,  "fp16_tflops": 125, "sm_count": 72},
        "L4":   {"bw_gb_s": 300,  "fp16_tflops": 121, "sm_count": 58},
        "T4":   {"bw_gb_s": 300,  "fp16_tflops": 65,  "sm_count": 40},
        "V100": {"bw_gb_s": 900,  "fp16_tflops": 125, "sm_count": 80},
        "3090": {"bw_gb_s": 936,  "fp16_tflops": 142, "sm_count": 82},
        "4090": {"bw_gb_s": 1008, "fp16_tflops": 330, "sm_count": 128},
    }

    def __init__(self):
        self.gpu_name = self._detect_gpu()
        self.specs = self.PEAK_SPECS.get(self.gpu_name, self.PEAK_SPECS["A100"])

    def _detect_gpu(self) -> str:
        if not torch.cuda.is_available():
            return "A100"
        name = torch.cuda.get_device_name(0).upper()
        for k in self.PEAK_SPECS:
            if k in name:
                return k
        return "A100"

    def bandwidth_utilization(
        self, bytes_accessed: int, elapsed_ms: float
    ) -> float:
        """Achieved BW as % of peak."""
        achieved = bytes_accessed / (elapsed_ms * 1e-3) / 1e9  # GB/s
        return achieved / self.specs["bw_gb_s"] * 100

    def flop_utilization(self, flops: int, elapsed_ms: float) -> float:
        """Achieved TFLOPS as % of peak FP16."""
        achieved = flops / (elapsed_ms * 1e-3) / 1e12  # TFLOPS
        return achieved / self.specs["fp16_tflops"] * 100

    def analyze_kernel(
        self,
        kernel_name: str,
        elapsed_ms: float,
        bytes_accessed: int,
        flops: int,
    ) -> Dict:
        """Full analysis of a single kernel's efficiency."""
        ai = flops / max(bytes_accessed, 1)  # arithmetic intensity
        ridge = self.specs["fp16_tflops"] * 1e12 / (self.specs["bw_gb_s"] * 1e9)

        regime = "memory-bound" if ai < ridge else "compute-bound"
        bw_util  = self.bandwidth_utilization(bytes_accessed, elapsed_ms)
        fp_util  = self.flop_utilization(flops, elapsed_ms)

        return {
            "kernel": kernel_name,
            "elapsed_ms": round(elapsed_ms, 3),
            "arithmetic_intensity": round(ai, 2),
            "ridge_point_flops_per_byte": round(ridge, 1),
            "regime": regime,
            "bandwidth_utilization_pct": round(bw_util, 1),
            "flop_utilization_pct": round(fp_util, 1),
            "bottleneck": "memory BW" if regime == "memory-bound" else "compute",
            "gpu": self.gpu_name,
        }

    def print_roofline_summary(self, analyses: List[Dict]):
        """Print a text roofline chart."""
        print("\n" + "="*65)
        print(f"ROOFLINE ANALYSIS — GPU: {self.gpu_name}")
        print(f"Peak BW: {self.specs['bw_gb_s']} GB/s | "
              f"Peak FP16: {self.specs['fp16_tflops']} TFLOPS")
        print("="*65)
        print(f"{'Kernel':<30} {'AI':>6} {'BW%':>6} {'FLOP%':>7} {'Regime'}")
        print("-"*65)
        for a in analyses:
            print(f"{a['kernel']:<30} {a['arithmetic_intensity']:>6.2f} "
                  f"{a['bandwidth_utilization_pct']:>5.1f}% "
                  f"{a['flop_utilization_pct']:>6.1f}%  {a['regime']}")
        print("="*65)
        print("\nNote: Our kernels are memory-bound (expected for small MLPs).")
        print("Triton fusion wins by reducing HBM round-trips, not FLOPS.")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def profile_section(name: str, profiler: Optional[UncertaintyProfiler] = None):
    """
    Context manager for labeling a section in the profiler trace.
    Works with both torch.profiler and nsight (via NVTX ranges).

    Usage:
        with profile_section("uncertainty_head_forward"):
            uncertainty = head(hidden_states)
    """
    if profiler and PROFILER_AVAILABLE:
        with record_function(name):
            try:
                # Also push NVTX range for nsight Systems visualization
                torch.cuda.nvtx.range_push(name)
                yield
            finally:
                torch.cuda.nvtx.range_pop()
    else:
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
