"""
benchmarks/bench_ablation.py

Ablation study for UncertaintyDecode hyperparameters.
Produces Section 4.6 of the paper: how sensitive is the system
to each design choice?

Ablations:
  1. uncertainty_weight α: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
  2. Block aggregation: max, mean, p75, p90
  3. Block size: 8, 16, 32 tokens/block
  4. Uncertainty threshold θ: 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8

Each ablation runs TruthfulQA (200 samples) + measures TTFT overhead.
Total runtime: ~2 hours on A100.

Run:
    python benchmarks/bench_ablation.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output results/ablation.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class AblationResult:
    ablation_name: str
    ablation_value: Any
    mc1_accuracy: float
    hallucination_rate: float
    ttft_ms: float
    protection_rate: float  # fraction of blocks protected per request


def run_single_config(
    model_name: str,
    n_samples: int,
    uncertainty_weight: float = 0.4,
    uncertainty_threshold: float = 0.65,
    block_aggregation: str = "max",
    kv_budget: float = 0.6,
) -> Dict:
    """Run TruthfulQA eval with a specific config. Returns metrics dict."""
    from uncertainty_decode.serving.llm import UncertaintyDecodeLLM
    from uncertainty_decode.eviction.block_scorer import BlockScorer
    from vllm import SamplingParams
    from datasets import load_dataset
    import time

    # Patch BlockScorer aggregation
    BlockScorer.__init__.__defaults__ = (16, block_aggregation, uncertainty_threshold)

    llm = UncertaintyDecodeLLM(
        model=model_name,
        uncertainty_threshold=uncertainty_threshold,
        kv_budget=kv_budget,
        uncertainty_weight=uncertainty_weight,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=100)

    dataset = list(load_dataset("truthful_qa", "multiple_choice", split="validation"))[:n_samples]

    mc1_scores = []
    halluc_labels = []
    ttft_times = []
    protection_rates = []

    for i in range(0, len(dataset), 8):
        batch = dataset[i:i+8]
        prompts = []
        for s in batch:
            choices_str = "\n".join(
                f"{chr(65+j)}. {c}" for j, c in enumerate(s["mc1_targets"]["choices"])
            )
            prompts.append(
                f"Question: {s['question']}\n\nChoices:\n{choices_str}\n\nAnswer:"
            )

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000 / len(batch)
        ttft_times.append(ttft_ms)

        # Collect protection rate
        dist = llm.get_uncertainty_distribution()
        if dist:
            protection_rates.append(dist.get("pct_above_threshold", 0))

        for out, s in zip(outputs, batch):
            answer = out.outputs[0].text.strip()[:1].upper()
            correct_idx = s["mc1_targets"]["labels"].index(1)
            correct_letter = chr(65 + correct_idx)
            mc1 = int(answer == correct_letter)
            mc1_scores.append(mc1)
            halluc_labels.append(1 - mc1)

    return {
        "mc1_accuracy": float(np.mean(mc1_scores)),
        "hallucination_rate": float(np.mean(halluc_labels)),
        "ttft_ms": float(np.mean(ttft_times)),
        "protection_rate": float(np.mean(protection_rates)) if protection_rates else 0.0,
    }


def run_ablations(model_name: str, n_samples: int) -> List[AblationResult]:
    results = []

    # ── Ablation 1: uncertainty_weight α ────────────────────────────────────
    print("\n" + "="*50)
    print("Ablation 1: uncertainty_weight α")
    print("="*50)
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"  α = {alpha}...")
        metrics = run_single_config(model_name, n_samples, uncertainty_weight=alpha)
        results.append(AblationResult(
            ablation_name="uncertainty_weight",
            ablation_value=alpha,
            **metrics,
        ))
        print(f"    MC1={metrics['mc1_accuracy']:.3f}, "
              f"Halluc={metrics['hallucination_rate']:.3f}, "
              f"TTFT={metrics['ttft_ms']:.1f}ms")

    # ── Ablation 2: Block aggregation strategy ───────────────────────────────
    print("\n" + "="*50)
    print("Ablation 2: Block aggregation strategy")
    print("="*50)
    for agg in ["max", "mean", "p75", "p90"]:
        print(f"  agg = {agg}...")
        metrics = run_single_config(model_name, n_samples, block_aggregation=agg)
        results.append(AblationResult(
            ablation_name="block_aggregation",
            ablation_value=agg,
            **metrics,
        ))
        print(f"    MC1={metrics['mc1_accuracy']:.3f}")

    # ── Ablation 3: Uncertainty threshold θ ─────────────────────────────────
    print("\n" + "="*50)
    print("Ablation 3: Uncertainty threshold θ")
    print("="*50)
    for theta in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        print(f"  θ = {theta}...")
        metrics = run_single_config(
            model_name, n_samples, uncertainty_threshold=theta
        )
        results.append(AblationResult(
            ablation_name="uncertainty_threshold",
            ablation_value=theta,
            **metrics,
        ))
        print(
            f"    MC1={metrics['mc1_accuracy']:.3f}, "
            f"protected={metrics['protection_rate']:.1%}"
        )

    return results


def print_ablation_tables(results: List[AblationResult]) -> None:
    """Print one table per ablation dimension."""

    for ablation_name in ["uncertainty_weight", "block_aggregation", "uncertainty_threshold"]:
        subset = [r for r in results if r.ablation_name == ablation_name]
        if not subset:
            continue

        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"{'='*60}")
        print(f"{'Value':<15} {'MC1 Acc':<10} {'Halluc%':<10} "
              f"{'TTFT(ms)':<12} {'Prot%'}")
        print("-"*60)
        for r in subset:
            print(
                f"{str(r.ablation_value):<15} {r.mc1_accuracy:<10.3f} "
                f"{r.hallucination_rate:<10.3f} {r.ttft_ms:<12.1f} "
                f"{r.protection_rate:.1%}"
            )

        # Find best
        best = max(subset, key=lambda r: r.mc1_accuracy)
        print(f"\n  Best {ablation_name}: {best.ablation_value} "
              f"(MC1={best.mc1_accuracy:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output", default="results/ablation.json")
    args = parser.parse_args()

    results = run_ablations(args.model, args.n_samples)
    print_ablation_tables(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nAblation results saved to {args.output}")


if __name__ == "__main__":
    main()
