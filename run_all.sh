#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Reproduce all UncertaintyDecode paper results
#
# Runtime estimate:
#   A100 40GB: ~4 hours total
#   T4 16GB:   ~12 hours total (reduce --n-samples to speed up)
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh                          # full run
#   ./run_all.sh --fast                   # quick sanity check (50 samples)
#   ./run_all.sh --model mistralai/Mistral-7B-Instruct-v0.2  # open model
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on error
set -u  # treat unset variables as errors

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
N_SAMPLES=200
KV_BUDGET=0.6
FAST=false
RESULTS_DIR="results"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)      FAST=true; N_SAMPLES=50; shift ;;
        --model)     MODEL="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --results)   RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          UncertaintyDecode — Full Evaluation Suite          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:     $MODEL"
echo "║  Samples:   $N_SAMPLES per eval"
echo "║  KV budget: $KV_BUDGET"
echo "║  Fast mode: $FAST"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "Unknown")
echo "GPU: $GPU_NAME"
echo ""

# ── Step 0: Run tests ─────────────────────────────────────────────────────────
echo "━━━ Step 0: Unit Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m pytest tests/ -v --tb=short -q 2>&1 | tee "$RESULTS_DIR/test_output.txt"
echo ""

# ── Step 1: Train uncertainty head ────────────────────────────────────────────
echo "━━━ Step 1: Train Uncertainty Head ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CHECKPOINT="checkpoints/uncertainty_head_$(basename $MODEL).pt"
if [ -f "$CHECKPOINT" ]; then
    echo "  Checkpoint found: $CHECKPOINT (skipping training)"
else
    echo "  Training uncertainty head on TriviaQA..."
    TRAIN_SAMPLES=$([[ "$FAST" == "true" ]] && echo 500 || echo 5000)
    python scripts/train_uncertainty_head.py \
        --model "$MODEL" \
        --n-samples "$TRAIN_SAMPLES" \
        --epochs 3 \
        --output "$CHECKPOINT" \
        2>&1 | tee "$RESULTS_DIR/training.log"
    echo "  ✓ Checkpoint saved: $CHECKPOINT"
fi
echo ""

# ── Step 2: Triton kernel roofline benchmark ──────────────────────────────────
echo "━━━ Step 2: Triton Kernel Roofline Analysis ━━━━━━━━━━━━━━━━━━"
python benchmarks/bench_roofline.py \
    --output "$RESULTS_DIR/roofline" \
    2>&1 | tee "$RESULTS_DIR/roofline.log"
echo ""

# ── Step 3: Latency benchmark (TTFT / TPOT / throughput) ─────────────────────
echo "━━━ Step 3: Latency Benchmark ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
N_REQUESTS=$([[ "$FAST" == "true" ]] && echo 50 || echo 200)
python benchmarks/bench_latency.py \
    --model "$MODEL" \
    --n-requests "$N_REQUESTS" \
    --kv-budget "$KV_BUDGET" \
    --policies lru h2o uncertainty_decode \
    --output "$RESULTS_DIR/latency_bench.json" \
    2>&1 | tee "$RESULTS_DIR/latency.log"
echo ""

# ── Step 4: TruthfulQA evaluation ────────────────────────────────────────────
echo "━━━ Step 4: TruthfulQA Hallucination Evaluation ━━━━━━━━━━━━━━"
python evals/eval_truthfulqa.py \
    --model "$MODEL" \
    --n-samples "$N_SAMPLES" \
    --kv-budget "$KV_BUDGET" \
    --policies lru h2o uncertainty_decode \
    --output "$RESULTS_DIR/truthfulqa_eval.json" \
    2>&1 | tee "$RESULTS_DIR/truthfulqa.log"
echo ""

# ── Step 5: LongBench evaluation ──────────────────────────────────────────────
echo "━━━ Step 5: LongBench Long-Context Evaluation ━━━━━━━━━━━━━━━━"
if [[ "$FAST" == "true" ]]; then
    TASKS="hotpotqa 2wikimqa"
    KV_BUDGETS="0.6"
else
    TASKS="narrativeqa qasper hotpotqa 2wikimqa musique"
    KV_BUDGETS="0.4 0.6 0.8"
fi

python evals/eval_longbench.py \
    --model "$MODEL" \
    --tasks $TASKS \
    --policies lru h2o uncertainty_decode \
    --kv-budgets $KV_BUDGETS \
    --n-samples "$N_SAMPLES" \
    --output "$RESULTS_DIR/longbench_eval.json" \
    2>&1 | tee "$RESULTS_DIR/longbench.log"
echo ""

# ── Step 6: Ablation study ────────────────────────────────────────────────────
if [[ "$FAST" == "false" ]]; then
    echo "━━━ Step 6: Ablation Study ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python benchmarks/bench_ablation.py \
        --model "$MODEL" \
        --n-samples "$N_SAMPLES" \
        --output "$RESULTS_DIR/ablation.json" \
        2>&1 | tee "$RESULTS_DIR/ablation.log"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ All evaluations complete!"
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json 2>/dev/null || true
echo ""
echo "Key files for the paper:"
echo "  $RESULTS_DIR/roofline/roofline_data.json   → Section 3.3 (kernel overhead)"
echo "  $RESULTS_DIR/latency_bench.json             → Table 3 (latency)"
echo "  $RESULTS_DIR/truthfulqa_eval.json           → Table 1 (hallucination rate)"
echo "  $RESULTS_DIR/longbench_eval.json            → Table 2 (long-context)"
echo "  $RESULTS_DIR/ablation.json                  → Table 4 (ablation)"
echo ""
echo "Next: fill in paper_outline.md with these numbers and submit to arXiv!"
