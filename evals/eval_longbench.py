"""
evals/eval_longbench.py

Evaluate long-context accuracy on LongBench benchmark.
This is the second core eval table in the paper.

LongBench tasks (we use 6):
  - NarrativeQA    (story comprehension, ~18K token avg)
  - Qasper          (scientific QA, ~12K)
  - MultiFieldQA    (multi-domain QA, ~11K)
  - HotpotQA        (multi-hop reasoning, ~9K)
  - 2WikiMultihopQA (multi-hop, ~6K)
  - MuSiQue         (multi-hop, ~11K)

These are specifically chosen because long contexts stress KV compression:
evicting the wrong tokens breaks multi-hop reasoning chains.

Run:
    python evals/eval_longbench.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --tasks narrativeqa qasper hotpotqa \
        --n-samples 100 \
        --kv-budgets 0.4 0.6 0.8 \
        --output results/longbench_eval.json
"""

import argparse
import json
import re
import string
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions (from the LongBench paper)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return int(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def score_answer(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    """Returns (f1, exact_match) taking the max over all ground truths."""
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    em = max(exact_match_score(prediction, gt) for gt in ground_truths)
    return f1, em


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading and prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "narrativeqa": {
        "dataset": "THUDM/LongBench",
        "subset": "narrativeqa",
        "prompt_template": "You are given a story. Answer the question based on the story.\n\nStory:\n{context}\n\nQuestion: {input}\nAnswer:",
        "max_tokens": 100,
    },
    "qasper": {
        "dataset": "THUDM/LongBench",
        "subset": "qasper",
        "prompt_template": "You are given a scientific paper. Answer the question based on the paper.\n\nPaper:\n{context}\n\nQuestion: {input}\nAnswer:",
        "max_tokens": 100,
    },
    "multifieldqa_en": {
        "dataset": "THUDM/LongBench",
        "subset": "multifieldqa_en",
        "prompt_template": "Read the document and answer the question.\n\nDocument:\n{context}\n\nQuestion: {input}\nAnswer:",
        "max_tokens": 50,
    },
    "hotpotqa": {
        "dataset": "THUDM/LongBench",
        "subset": "hotpotqa",
        "prompt_template": "Answer the question based on the given passages.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:",
        "max_tokens": 50,
    },
    "2wikimqa": {
        "dataset": "THUDM/LongBench",
        "subset": "2wikimqa",
        "prompt_template": "Answer the question based on the given document.\n\nDocument:\n{context}\n\nQuestion: {input}\nAnswer:",
        "max_tokens": 50,
    },
    "musique": {
        "dataset": "THUDM/LongBench",
        "subset": "musique",
        "prompt_template": "Answer the multi-hop question based on the paragraphs.\n\nParagraphs:\n{context}\n\nQuestion: {input}\nAnswer:",
        "max_tokens": 50,
    },
}


def load_longbench_task(task_name: str, n_samples: int) -> List[Dict]:
    if not DATASETS_AVAILABLE:
        raise ImportError("pip install datasets")

    config = TASK_CONFIGS[task_name]
    dataset = load_dataset(
        config["dataset"],
        config["subset"],
        split="test",
        trust_remote_code=True,
    )
    return list(dataset)[:n_samples]


def format_prompt(sample: Dict, task_name: str, max_context_chars: int = 12000) -> str:
    """Format a LongBench sample into a prompt, truncating context if needed."""
    config = TASK_CONFIGS[task_name]
    context = sample.get("context", "")[:max_context_chars]
    input_text = sample.get("input", "")
    return config["prompt_template"].format(context=context, input=input_text)


def get_ground_truths(sample: Dict) -> List[str]:
    answers = sample.get("answers", [])
    if isinstance(answers, str):
        return [answers]
    return answers if answers else [sample.get("answer", "")]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task: str
    policy: str
    kv_budget: float
    n_samples: int
    f1: float
    exact_match: float
    avg_seq_len: float


@dataclass
class LongBenchResult:
    policy: str
    model: str
    kv_budget: float
    n_samples_per_task: int
    task_results: List[TaskResult]
    macro_avg_f1: float
    macro_avg_em: float


def evaluate_task(
    task_name: str,
    policy_name: str,
    model_name: str,
    samples: List[Dict],
    kv_budget: float,
    uncertainty_threshold: float,
    uncertainty_weight: float,
) -> TaskResult:
    """Evaluate one policy on one LongBench task."""

    config = TASK_CONFIGS[task_name]

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

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config["max_tokens"],
        stop=["\n", "Question:"],
    )

    prompts = [format_prompt(s, task_name) for s in samples]
    ground_truths = [get_ground_truths(s) for s in samples]

    # Track sequence lengths for the paper
    seq_lens = [len(p.split()) for p in prompts]

    f1_scores = []
    em_scores = []

    batch_size = 4
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_gts = ground_truths[i:i+batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for out, gts in zip(outputs, batch_gts):
            prediction = out.outputs[0].text.strip()
            f1, em = score_answer(prediction, gts)
            f1_scores.append(f1)
            em_scores.append(em)

    return TaskResult(
        task=task_name,
        policy=policy_name,
        kv_budget=kv_budget,
        n_samples=len(samples),
        f1=float(np.mean(f1_scores)),
        exact_match=float(np.mean(em_scores)),
        avg_seq_len=float(np.mean(seq_lens)),
    )


def run_longbench_eval(
    model_name: str,
    task_names: List[str],
    policy_names: List[str],
    kv_budgets: List[float],
    n_samples: int,
    uncertainty_threshold: float = 0.65,
    uncertainty_weight: float = 0.4,
) -> List[LongBenchResult]:
    """Full LongBench evaluation across all policies and KV budgets."""

    # Pre-load all datasets
    print("Loading LongBench datasets...")
    task_samples = {
        task: load_longbench_task(task, n_samples)
        for task in task_names
    }

    all_results = []

    for kv_budget in kv_budgets:
        for policy_name in policy_names:
            print(f"\n{'='*60}")
            print(f"Policy: {policy_name} | KV budget: {kv_budget}")
            print(f"{'='*60}")

            task_results = []
            for task_name in task_names:
                print(f"\n  Task: {task_name}")
                result = evaluate_task(
                    task_name=task_name,
                    policy_name=policy_name,
                    model_name=model_name,
                    samples=task_samples[task_name],
                    kv_budget=kv_budget,
                    uncertainty_threshold=uncertainty_threshold,
                    uncertainty_weight=uncertainty_weight,
                )
                task_results.append(result)
                print(f"    F1: {result.f1:.3f} | EM: {result.exact_match:.3f}")

            lb_result = LongBenchResult(
                policy=policy_name,
                model=model_name.split("/")[-1],
                kv_budget=kv_budget,
                n_samples_per_task=n_samples,
                task_results=task_results,
                macro_avg_f1=float(np.mean([r.f1 for r in task_results])),
                macro_avg_em=float(np.mean([r.exact_match for r in task_results])),
            )
            all_results.append(lb_result)

    return all_results


def print_longbench_table(results: List[LongBenchResult], task_names: List[str]) -> None:
    """Print LaTeX-style table for the paper."""
    print("\n" + "="*90)
    print("LONGBENCH RESULTS (F1)")
    print("="*90)

    # Header
    task_abbrevs = {
        "narrativeqa": "NrrQA", "qasper": "Qaspr",
        "multifieldqa_en": "MFldQA", "hotpotqa": "HpQA",
        "2wikimqa": "2Wiki", "musique": "MuSiQ",
    }
    header_tasks = " ".join(f"{task_abbrevs.get(t, t[:6]):<8}" for t in task_names)
    print(f"{'Policy':<22} {'Budget':<8} {header_tasks} {'Avg F1'}")
    print("-"*90)

    for r in results:
        task_f1s = {tr.task: tr.f1 for tr in r.task_results}
        f1_str = " ".join(f"{task_f1s.get(t, 0):<8.3f}" for t in task_names)
        print(f"{r.policy:<22} {r.kv_budget:<8.1f} {f1_str} {r.macro_avg_f1:.3f}")

    print("="*90)

    # Highlight best per budget
    print("\nBest policy per KV budget:")
    budgets = sorted(set(r.kv_budget for r in results))
    for budget in budgets:
        budget_results = [r for r in results if r.kv_budget == budget]
        best = max(budget_results, key=lambda r: r.macro_avg_f1)
        print(f"  Budget {budget:.0%}: {best.policy} (F1={best.macro_avg_f1:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--tasks", nargs="+",
                        default=["narrativeqa", "qasper", "hotpotqa", "2wikimqa", "musique"])
    parser.add_argument("--policies", nargs="+",
                        default=["lru", "h2o", "uncertainty_decode"])
    parser.add_argument("--kv-budgets", type=float, nargs="+",
                        default=[0.4, 0.6, 0.8])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.65)
    parser.add_argument("--uncertainty-weight", type=float, default=0.4)
    parser.add_argument("--output", default="results/longbench_eval.json")
    args = parser.parse_args()

    results = run_longbench_eval(
        model_name=args.model,
        task_names=args.tasks,
        policy_names=args.policies,
        kv_budgets=args.kv_budgets,
        n_samples=args.n_samples,
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_weight=args.uncertainty_weight,
    )

    print_longbench_table(results, args.tasks)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Serialize (task_results are dataclasses too)
    output_data = []
    for r in results:
        rd = asdict(r)
        output_data.append(rd)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    from dataclasses import asdict
    main()
