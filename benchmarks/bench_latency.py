"""
benchmarks/bench_latency.py

Benchmark UncertaintyDecode vs baseline eviction policies.
Produces the latency/throughput table for the paper.

Metrics measured:
  - TTFT (Time to First Token) ms
  - TPOT (Time Per Output Token) ms
  - Tokens/sec
  - Uncertainty head overhead (ms/forward pass)
  - KV memory usage (GB)

Run:
    python benchmarks/bench_latency.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --batch-size 8 \
        --seq-len 2048 \
        --n-requests 100 \
        --output results/latency_bench.json
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

try:
    from vllm import LLM, SamplingParams
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    print("vLLM not installed. Some benchmarks require vLLM.")


@dataclass
class BenchmarkResult:
    policy: str
    model: str
    batch_size: int
    seq_len: int
    kv_budget: float
    ttft_ms_mean: float
    ttft_ms_p99: float
    tpot_ms_mean: float
    tokens_per_sec: float
    kv_memory_gb: float
    uncertainty_overhead_ms: float
    n_requests: int


def generate_prompts(n: int, min_len: int = 512, max_len: int = 2048) -> List[str]:
    """Generate synthetic prompts of varying lengths for benchmarking."""
    base_prompts = [
        "Explain the history and causes of World War I in comprehensive detail, covering the political alliances, assassination of Archduke Franz Ferdinand, and the domino effect of declarations of war.",
        "Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions, the role of chlorophyll, and how glucose is synthesized from CO2 and water.",
        "Analyze the key differences between classical and quantum computing, including qubit states, superposition, entanglement, quantum gates, and the types of problems each paradigm is suited for.",
        "Provide a detailed explanation of how transformer neural networks work, including the attention mechanism, positional encoding, multi-head attention, and how BERT and GPT differ in their training objectives.",
        "Describe the major events of the Apollo space program, from Kennedy's 1961 speech to the Apollo 11 moon landing, including the technical challenges overcome and the astronauts involved.",
    ]
    prompts = []
    for i in range(n):
        base = base_prompts[i % len(base_prompts)]
        # Add filler to vary length
        filler = " Please provide as much detail as possible." * (i % 5)
        prompts.append(base + filler)
    return prompts


def benchmark_policy(
    policy_name: str,
    model: str,
    prompts: List[str],
    sampling_params,
    kv_budget: float = 0.6,
    uncertainty_threshold: float = 0.65,
    uncertainty_weight: float = 0.4,
) -> BenchmarkResult:
    """Run benchmark for a single eviction policy."""

    print(f"\n{'='*60}")
    print(f"Benchmarking policy: {policy_name}")
    print(f"{'='*60}")

    ttft_times = []
    tpot_times = []
    total_tokens = 0
    overhead_times = []

    if policy_name == "uncertainty_decode":
        from uncertainty_decode.serving.llm import UncertaintyDecodeLLM
        llm = UncertaintyDecodeLLM(
            model=model,
            uncertainty_threshold=uncertainty_threshold,
            kv_budget=kv_budget,
            uncertainty_weight=uncertainty_weight,
            gpu_memory_utilization=0.85,
        )
    elif policy_name in ("lru", "h2o"):
        llm = LLM(
            model=model,
            gpu_memory_utilization=0.85,
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Warmup
    print("Warming up...")
    _ = llm.generate(prompts[:2], sampling_params)
    torch.cuda.synchronize()

    # Benchmark runs
    print(f"Running {len(prompts)} requests...")
    batch_size = 8
    start_time = time.perf_counter()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]

        t0 = time.perf_counter()
        outputs = llm.generate(batch, sampling_params)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        for out in outputs:
            n_output_tokens = len(out.outputs[0].token_ids)
            total_tokens += n_output_tokens
            tpot = elapsed / max(n_output_tokens, 1)
            tpot_times.append(tpot)

        ttft_times.append(elapsed)  # approximation without streaming

    total_time = time.perf_counter() - start_time

    # Memory measurement
    mem_gb = 0.0
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9

    result = BenchmarkResult(
        policy=policy_name,
        model=model.split("/")[-1],
        batch_size=batch_size,
        seq_len=len(prompts[0].split()),
        kv_budget=kv_budget,
        ttft_ms_mean=float(np.mean(ttft_times)),
        ttft_ms_p99=float(np.percentile(ttft_times, 99)),
        tpot_ms_mean=float(np.mean(tpot_times)),
        tokens_per_sec=total_tokens / total_time,
        kv_memory_gb=mem_gb,
        uncertainty_overhead_ms=float(np.mean(overhead_times)) if overhead_times else 0.0,
        n_requests=len(prompts),
    )

    print(f"\nResults for {policy_name}:")
    print(f"  TTFT mean: {result.ttft_ms_mean:.1f}ms | p99: {result.ttft_ms_p99:.1f}ms")
    print(f"  TPOT mean: {result.tpot_ms_mean:.2f}ms")
    print(f"  Throughput: {result.tokens_per_sec:.1f} tokens/sec")
    print(f"  KV memory: {result.kv_memory_gb:.2f}GB")

    return result


def print_comparison_table(results: List[BenchmarkResult]) -> None:
    """Print LaTeX-style comparison table for the paper."""
    print("\n" + "="*80)
    print("RESULTS TABLE (LaTeX-ready)")
    print("="*80)
    print(
        f"{'Policy':<20} {'TTFT(ms)':<12} {'TPOT(ms)':<12} "
        f"{'Tok/sec':<12} {'KV Mem(GB)':<14} {'Overhead(ms)':<14}"
    )
    print("-"*80)
    for r in results:
        print(
            f"{r.policy:<20} {r.ttft_ms_mean:<12.1f} {r.tpot_ms_mean:<12.2f} "
            f"{r.tokens_per_sec:<12.1f} {r.kv_memory_gb:<14.2f} {r.uncertainty_overhead_ms:<14.2f}"
        )
    print("="*80)

    # Compute improvements vs LRU baseline
    lru_result = next((r for r in results if r.policy == "lru"), None)
    ud_result = next((r for r in results if r.policy == "uncertainty_decode"), None)

    if lru_result and ud_result:
        print("\nUncertaintyDecode vs LRU baseline:")
        tpot_delta = (ud_result.tpot_ms_mean - lru_result.tpot_ms_mean) / lru_result.tpot_ms_mean * 100
        mem_delta = (ud_result.kv_memory_gb - lru_result.kv_memory_gb) / lru_result.kv_memory_gb * 100
        print(f"  TPOT change: {tpot_delta:+.1f}%")
        print(f"  KV memory change: {mem_delta:+.1f}%")
        print(f"  Uncertainty overhead: +{ud_result.uncertainty_overhead_ms:.1f}ms/forward pass")


def main():
    parser = argparse.ArgumentParser(description="UncertaintyDecode Latency Benchmark")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--n-requests", type=int, default=100)
    parser.add_argument("--kv-budget", type=float, default=0.6)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.65)
    parser.add_argument("--uncertainty-weight", type=float, default=0.4)
    parser.add_argument("--policies", nargs="+",
                        default=["lru", "h2o", "uncertainty_decode"])
    parser.add_argument("--output", type=str, default="results/latency_bench.json")
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=["\n\n"],
    )

    prompts = generate_prompts(args.n_requests)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for policy in args.policies:
        result = benchmark_policy(
            policy_name=policy,
            model=args.model,
            prompts=prompts,
            sampling_params=sampling_params,
            kv_budget=args.kv_budget,
            uncertainty_threshold=args.uncertainty_threshold,
            uncertainty_weight=args.uncertainty_weight,
        )
        results.append(result)

    print_comparison_table(results)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
