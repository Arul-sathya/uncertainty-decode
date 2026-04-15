"""
evals/eval_truthfulqa.py

Evaluate hallucination rate on TruthfulQA benchmark.
Compares UncertaintyDecode vs LRU and H2O baselines.

Produces:
  - Accuracy on TruthfulQA MC1 and MC2
  - AUROC of uncertainty scores as hallucination predictors
  - Per-category breakdown

Usage:
    python evals/eval_truthfulqa.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --n-samples 200 \
        --output results/truthfulqa_eval.json

Requires: datasets (pip install datasets)
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EvalResult:
    policy: str
    model: str
    n_samples: int
    mc1_accuracy: float       # multiple choice, 1 correct answer
    mc2_accuracy: float       # multiple choice, multiple correct
    hallucination_rate: float # fraction of clearly wrong answers
    uncertainty_auroc: float  # AUROC of uncertainty as hallucination predictor
    kv_budget: float


def load_truthfulqa(n_samples: int = 200) -> List[Dict]:
    """Load TruthfulQA dataset."""
    if not DATASETS_AVAILABLE:
        raise ImportError("pip install datasets")

    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    samples = list(dataset)[:n_samples]
    return samples


def score_mc1(model_answer: str, correct_answer: str) -> int:
    """Score multiple choice 1 (single best answer)."""
    return int(model_answer.strip().lower() == correct_answer.strip().lower())


def evaluate_policy(
    policy_name: str,
    model_name: str,
    samples: List[Dict],
    kv_budget: float = 0.6,
    uncertainty_threshold: float = 0.65,
    uncertainty_weight: float = 0.4,
) -> EvalResult:
    """Evaluate a single eviction policy on TruthfulQA."""

    print(f"\nEvaluating {policy_name} on {len(samples)} TruthfulQA samples...")

    if policy_name == "uncertainty_decode":
        from uncertainty_decode.serving.llm import UncertaintyDecodeLLM
        from vllm import SamplingParams
        llm = UncertaintyDecodeLLM(
            model=model_name,
            uncertainty_threshold=uncertainty_threshold,
            kv_budget=kv_budget,
            uncertainty_weight=uncertainty_weight,
        )
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_name, gpu_memory_utilization=0.85)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    mc1_scores = []
    mc2_scores = []
    uncertainty_scores_all = []
    hallucination_labels = []

    batch_size = 8
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]

        # Format prompts as multiple choice
        prompts = []
        for s in batch:
            q = s["question"]
            choices = s["mc1_targets"]["choices"]
            choices_str = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
            prompt = (
                f"Question: {q}\n\n"
                f"Choices:\n{choices_str}\n\n"
                f"Answer with just the letter (A, B, C, etc.):"
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

        # Capture uncertainty scores if available
        if policy_name == "uncertainty_decode":
            scores = llm.get_last_uncertainty_scores()
            if scores is not None:
                # Use mean uncertainty per sequence as hallucination predictor
                for j in range(min(len(batch), scores.shape[0])):
                    uncertainty_scores_all.append(float(scores[j].mean()))

        for j, (s, out) in enumerate(zip(batch, outputs)):
            answer = out.outputs[0].text.strip()[:1].upper()  # just the letter
            correct_idx = s["mc1_targets"]["labels"].index(1)
            correct_letter = chr(65 + correct_idx)

            mc1 = score_mc1(answer, correct_letter)
            mc1_scores.append(mc1)

            # Hallucination label: 1 if wrong, 0 if correct
            is_hallucinated = 1 - mc1
            hallucination_labels.append(is_hallucinated)

            # MC2: any correct answer
            correct_letters = [
                chr(65 + k)
                for k, label in enumerate(s["mc1_targets"]["labels"])
                if label == 1
            ]
            mc2 = int(answer in correct_letters)
            mc2_scores.append(mc2)

    # Compute AUROC of uncertainty as hallucination predictor
    auroc = 0.0
    if uncertainty_scores_all and SKLEARN_AVAILABLE and len(set(hallucination_labels)) > 1:
        auroc = roc_auc_score(hallucination_labels, uncertainty_scores_all)

    result = EvalResult(
        policy=policy_name,
        model=model_name.split("/")[-1],
        n_samples=len(samples),
        mc1_accuracy=float(np.mean(mc1_scores)),
        mc2_accuracy=float(np.mean(mc2_scores)),
        hallucination_rate=float(np.mean(hallucination_labels)),
        uncertainty_auroc=float(auroc),
        kv_budget=kv_budget,
    )

    print(f"  MC1 Accuracy:       {result.mc1_accuracy:.3f}")
    print(f"  MC2 Accuracy:       {result.mc2_accuracy:.3f}")
    print(f"  Hallucination Rate: {result.hallucination_rate:.3f}")
    print(f"  Uncertainty AUROC:  {result.uncertainty_auroc:.3f}")

    return result


def print_eval_table(results: List[EvalResult]) -> None:
    """Print comparison table for the paper."""
    print("\n" + "="*70)
    print("TRUTHFULQA EVALUATION RESULTS")
    print("="*70)
    print(
        f"{'Policy':<22} {'MC1 Acc':<10} {'Halluc%':<10} "
        f"{'Unc AUROC':<12} {'KV Budget'}"
    )
    print("-"*70)
    for r in results:
        print(
            f"{r.policy:<22} {r.mc1_accuracy:<10.3f} {r.hallucination_rate:<10.3f} "
            f"{r.uncertainty_auroc:<12.3f} {r.kv_budget}"
        )
    print("="*70)

    # Highlight improvement
    lru = next((r for r in results if r.policy == "lru"), None)
    ud = next((r for r in results if r.policy == "uncertainty_decode"), None)
    if lru and ud:
        mc1_delta = (ud.mc1_accuracy - lru.mc1_accuracy) / lru.mc1_accuracy * 100
        hall_delta = (ud.hallucination_rate - lru.hallucination_rate) / lru.hallucination_rate * 100
        print(f"\nUncertaintyDecode vs LRU:")
        print(f"  MC1 accuracy: {mc1_delta:+.1f}%")
        print(f"  Hallucination rate: {hall_delta:+.1f}%")
        print(f"  Uncertainty AUROC (hallucination prediction): {ud.uncertainty_auroc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--kv-budget", type=float, default=0.6)
    parser.add_argument("--policies", nargs="+",
                        default=["lru", "h2o", "uncertainty_decode"])
    parser.add_argument("--output", default="results/truthfulqa_eval.json")
    args = parser.parse_args()

    samples = load_truthfulqa(args.n_samples)

    results = []
    for policy in args.policies:
        result = evaluate_policy(
            policy_name=policy,
            model_name=args.model,
            samples=samples,
            kv_budget=args.kv_budget,
        )
        results.append(result)

    print_eval_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
