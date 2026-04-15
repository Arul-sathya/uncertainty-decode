"""
scripts/train_uncertainty_head.py

Train the Dirichlet uncertainty head on a small labeled dataset.
Runtime: ~30 minutes on A100, ~2 hours on T4 (free Colab).

The head learns to produce high uncertainty scores for tokens
that precede hallucinations and low scores for grounded tokens.

Training data construction:
  1. Run a base LLM on factual QA pairs (e.g. TriviaQA)
  2. Compare output to gold answer → binary hallucination label
  3. Extract hidden states from the final layer norm
  4. Train the head: hidden_state → Dirichlet params → uncertainty

Run:
    python scripts/train_uncertainty_head.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --n-samples 5000 \
        --epochs 3 \
        --output checkpoints/uncertainty_head.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

from uncertainty_decode.eviction.uncertainty_head import (
    DirichletEvidenceHead,
    UncertaintyConfig,
)


class HallucinationDataset(Dataset):
    """
    Dataset of (hidden_state, label) pairs for training the uncertainty head.

    hidden_state: [D] — hidden state of one token from the final layer norm
    label: int — 0 = grounded (no hallucination), 1 = hallucinated

    Construction: run your base model on TriviaQA/NQ, extract hidden states,
    compare generations to gold → assign labels.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,   # [N, D]
        labels: torch.Tensor,          # [N] — binary
    ):
        assert len(hidden_states) == len(labels)
        self.hidden_states = hidden_states
        self.labels = labels

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx]


def extract_hidden_states(
    model_name: str,
    n_samples: int = 5000,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the base model on TriviaQA and extracts (hidden_states, labels).
    Labels: 1 if the generated answer doesn't match gold, 0 otherwise.

    Returns:
        hidden_states: [N, D] float16
        labels: [N] int
    """
    print(f"Extracting hidden states from {model_name}...")

    try:
        from vllm import LLM, SamplingParams
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install vllm datasets")

    # Load a small factual QA dataset
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="train[:5000]")
    samples = list(dataset)[:n_samples]

    llm = LLM(model=model_name, gpu_memory_utilization=0.7)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    all_hidden = []
    all_labels = []

    # We'll use a forward hook to capture hidden states
    from uncertainty_decode.serving.llm import UncertaintyDecodeHook
    from uncertainty_decode.eviction.uncertainty_head import UncertaintyHeadRegistry
    from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

    captured_hidden = []

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_hidden.append(h.detach().cpu().float())

    # Register hook on final layer norm
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    hook_handle = None
    for name, module in model.named_modules():
        if "norm" in name.lower() and isinstance(module, nn.LayerNorm):
            hook_handle = module.register_forward_hook(capture_hook)
            print(f"Hook registered on: {name}")
            break

    # Generate answers and compare to gold
    prompts = [
        f"Answer this question with one word or phrase: {s['question']}"
        for s in samples
    ]

    batch_size = 16
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        captured_hidden.clear()
        batch_prompts = prompts[i:i+batch_size]
        batch_samples = samples[i:i+batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for j, (out, sample) in enumerate(zip(outputs, batch_samples)):
            gen_answer = out.outputs[0].text.strip().lower()
            gold_answers = [
                a.lower() for a in sample["answer"]["aliases"]
            ] + [sample["answer"]["value"].lower()]

            # Label: 1 = hallucination (wrong answer), 0 = grounded
            is_hallucinated = int(
                not any(gold in gen_answer or gen_answer in gold for gold in gold_answers)
            )

            # Use the captured hidden states for this sequence
            if j < len(captured_hidden):
                h = captured_hidden[j]  # [T, D]
                # Use last non-padding token's hidden state
                all_hidden.append(h[-1])  # [D]
                all_labels.append(is_hallucinated)

    if hook_handle:
        hook_handle.remove()

    hidden_tensor = torch.stack(all_hidden)          # [N, D]
    labels_tensor = torch.tensor(all_labels)         # [N]

    halluc_rate = labels_tensor.float().mean()
    print(f"Extracted {len(all_hidden)} samples, hallucination rate: {halluc_rate:.2%}")

    return hidden_tensor, labels_tensor


def train(
    model_name: str,
    n_samples: int,
    epochs: int,
    lr: float,
    batch_size: int,
    output_path: str,
    device: str = "cuda",
) -> None:
    """Full training loop for the Dirichlet uncertainty head."""

    # Extract training data
    hidden_states, labels = extract_hidden_states(model_name, n_samples, device)

    # Infer hidden size
    D = hidden_states.shape[1]
    print(f"Hidden size: {D}")

    # Split train/val
    n_val = max(100, int(0.1 * len(hidden_states)))
    perm = torch.randperm(len(hidden_states))
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_dataset = HallucinationDataset(hidden_states[train_idx], labels[train_idx])
    val_dataset = HallucinationDataset(hidden_states[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize head
    config = UncertaintyConfig(hidden_size=D, proj_size=256, num_classes=2)
    head = DirichletEvidenceHead(config).to(device).float()

    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    print(f"\nTraining uncertainty head: {sum(p.numel() for p in head.parameters()):,} params")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        head.train()
        train_loss = 0.0
        annealing = min(1.0, epoch / max(epochs - 1, 1))  # anneal KL weight

        for h_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            h_batch = h_batch.to(device)
            y_batch = y_batch.to(device)

            # Add sequence dim: [B, 1, D]
            h_seq = h_batch.unsqueeze(1)

            output = head(h_seq)
            evidence = output["evidence"].squeeze(1)  # [B, K]
            labels_tok = y_batch                      # [B]

            loss = head.compute_edl_loss(
                evidence.unsqueeze(1),
                labels_tok.unsqueeze(1),
                annealing_coeff=annealing,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        head.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for h_batch, y_batch in val_loader:
                h_batch = h_batch.to(device)
                y_batch = y_batch.to(device)

                h_seq = h_batch.unsqueeze(1)
                output = head(h_seq)
                evidence = output["evidence"].squeeze(1)
                uncertainty = output["uncertainty"].squeeze(1)  # [B]

                loss = head.compute_edl_loss(
                    evidence.unsqueeze(1),
                    y_batch.unsqueeze(1),
                    annealing_coeff=annealing,
                )
                val_loss += loss.item()

                # Accuracy: predict hallucination if uncertainty > threshold
                preds = (uncertainty > config.threshold).long()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        val_loss /= len(val_loader)
        val_acc = correct / total
        scheduler.step()

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  ✓ New best checkpoint")

    # Save best checkpoint
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "config": config.__dict__,
        "best_val_loss": best_val_loss,
        "model_name": model_name,
        "n_training_samples": n_samples,
    }, output_path)
    print(f"\nCheckpoint saved to {output_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="checkpoints/uncertainty_head.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        model_name=args.model,
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
