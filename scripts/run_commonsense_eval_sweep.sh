#!/bin/bash
# Evaluate all commonsense checkpoints across multiple GPUs in parallel.
#
# Usage:
#   N_GPUS=8 bash scripts/run_commonsense_eval_sweep.sh

set -euo pipefail

METHODS="${METHODS:-lora adalora eva fim_lora}"
SEEDS="${SEEDS:-42 1337 2024}"
RANK="${RANK:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_GPUS="${N_GPUS:-1}"
TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"

mkdir -p logs

echo "=== Commonsense Eval Sweep ==="
echo "Methods: $METHODS"
echo "Seeds:   $SEEDS"
echo "Rank:    $RANK"
echo "GPUs:    $N_GPUS"
echo ""

# Install lm-evaluation-harness if not present
if ! uv run python -c "import lm_eval" 2>/dev/null; then
  echo "Installing lm-evaluation-harness..."
  uv add lm-eval
fi

# Build job list, skipping already-evaluated checkpoints
JOBS=()
SKIPPED=0
for METHOD in $METHODS; do
  for SEED in $SEEDS; do
    CHECKPOINT="$OUTPUT_DIR/$METHOD/commonsense/r$RANK/seed$SEED/checkpoint-final"
    RESULT="$OUTPUT_DIR/$METHOD/commonsense/r$RANK/seed$SEED/results.json"
    if [ -f "$RESULT" ]; then
      echo "  [skip] $METHOD / seed$SEED (results.json exists)"
      SKIPPED=$((SKIPPED + 1))
    elif [ ! -d "$CHECKPOINT" ]; then
      echo "  [missing] $CHECKPOINT — skipping"
    else
      JOBS+=("$METHOD|$SEED")
    fi
  done
done

TOTAL=${#JOBS[@]}
echo "Skipped: $SKIPPED"
echo "To eval: $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
  echo "All evaluations already complete."
  exit 0
fi

eval_on_gpu() {
  local GPU_ID=$1
  shift
  local -a MY_JOBS=("$@")

  for JOB in "${MY_JOBS[@]}"; do
    IFS='|' read -r METHOD SEED <<< "$JOB"
    CHECKPOINT="$OUTPUT_DIR/$METHOD/commonsense/r$RANK/seed$SEED/checkpoint-final"
    RESULT_DIR="$OUTPUT_DIR/$METHOD/commonsense/r$RANK/seed$SEED"

    echo ">>> [GPU $GPU_ID] eval $METHOD / seed$SEED"
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python src/eval_commonsense.py \
      --checkpoint "$CHECKPOINT" \
      --base-model "$BASE_MODEL" \
      --output-dir "$RESULT_DIR" \
      --batch-size "$BATCH_SIZE" \
      --tasks "$TASKS" 2>&1
    echo ""
  done
}

if [ "$N_GPUS" -eq 1 ]; then
  eval_on_gpu 0 "${JOBS[@]}"
else
  declare -a GPU_JOBS
  for ((i=0; i<N_GPUS; i++)); do GPU_JOBS[$i]=""; done

  for ((j=0; j<TOTAL; j++)); do
    GPU=$((j % N_GPUS))
    GPU_JOBS[$GPU]+="${JOBS[$j]} "
  done

  PIDS=()
  for ((i=0; i<N_GPUS; i++)); do
    if [ -n "${GPU_JOBS[$i]}" ]; then
      read -ra MY_JOBS <<< "${GPU_JOBS[$i]}"
      eval_on_gpu "$i" "${MY_JOBS[@]}" &
      PIDS+=($!)
      echo "Started eval worker for GPU $i (PID ${PIDS[-1]})"
    fi
  done

  FAILED=0
  for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
  done

  if [ "$FAILED" -ne 0 ]; then
    echo "One or more eval workers failed — check logs above."
    exit 1
  fi
fi

echo "=== All evaluations complete ==="
