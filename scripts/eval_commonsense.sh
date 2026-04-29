#!/bin/bash
# Evaluate a saved LoRA checkpoint on 7 commonsense reasoning tasks
# using lm-evaluation-harness (standard for LLM PEFT papers).
#
# Usage:
#   CHECKPOINT=results/fim_lora/commonsense/r16/seed42/checkpoint-final \
#   bash scripts/eval_commonsense.sh

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the model path}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"

echo "=== Commonsense Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Tasks: $TASKS"
echo ""

# Install lm-evaluation-harness if not present
if ! uv run python -c "import lm_eval" 2>/dev/null; then
  echo "Installing lm-evaluation-harness..."
  uv add lm-eval
fi

uv run python -m lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},peft=${CHECKPOINT},dtype=bfloat16" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "${OUTPUT_DIR}/lm_eval_$(basename $CHECKPOINT)" \
  --log_samples

echo "=== Evaluation complete ==="
