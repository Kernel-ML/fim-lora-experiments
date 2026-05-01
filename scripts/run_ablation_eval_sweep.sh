#!/bin/bash
# Evaluate all FIM ablation checkpoints across multiple GPUs.
#
# Usage:
#   N_GPUS=8 bash scripts/run_ablation_eval_sweep.sh

set -euo pipefail

SEEDS="${SEEDS:-42 1337 2024}"
RANK="${RANK:-16}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_GPUS="${N_GPUS:-1}"
TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"

mkdir -p logs

# label|checkpoint_base
CONFIGS=(
  "fim_lora_rmin4|results/fim_lora_rmin4/fim_lora"
  "fim_lora_rmin8|results/fim_lora_rmin8/fim_lora"
  "fim_lora_nb32|results/fim_lora_nb32/fim_lora"
  "fim_lora_rmin4_nb32|results/fim_lora_rmin4_nb32/fim_lora"
)

echo "=== FIM Ablation Eval Sweep ==="
echo "GPUs: $N_GPUS"
echo ""

JOBS=()
SKIPPED=0
for CONFIG in "${CONFIGS[@]}"; do
  IFS='|' read -r LABEL BASE <<< "$CONFIG"
  for SEED in $SEEDS; do
    RESULT="$BASE/commonsense/r$RANK/seed$SEED/results.json"
    CKPT="$BASE/commonsense/r$RANK/seed$SEED/checkpoint-final"
    if [ -f "$RESULT" ]; then
      echo "  [skip] $LABEL / $SEED"
      SKIPPED=$((SKIPPED + 1))
    elif [ ! -d "$CKPT" ]; then
      echo "  [missing checkpoint] $LABEL / $SEED"
    else
      JOBS+=("$LABEL|$SEED|$BASE")
    fi
  done
done

TOTAL=${#JOBS[@]}
echo "Skipped: $SKIPPED"
echo "To eval: $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then echo "All done."; exit 0; fi

eval_on_gpu() {
  local GPU_ID=$1; shift
  local -a MY_JOBS=("$@")
  for JOB in "${MY_JOBS[@]}"; do
    IFS='|' read -r LABEL SEED BASE <<< "$JOB"
    CKPT="$BASE/commonsense/r$RANK/seed$SEED/checkpoint-final"
    RESULT_DIR="$BASE/commonsense/r$RANK/seed$SEED"
    echo ">>> [GPU $GPU_ID] eval $LABEL / $SEED"
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python src/eval_commonsense.py \
      --checkpoint "$CKPT" \
      --base-model "$BASE_MODEL" \
      --output-dir "$RESULT_DIR" \
      --batch-size "$BATCH_SIZE" \
      --tasks "$TASKS" 2>&1
    echo ""
  done
}

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
    echo "Started eval worker GPU $i (PID ${PIDS[-1]})"
  fi
done

FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
[ "$FAILED" -ne 0 ] && echo "One or more eval workers failed." && exit 1
echo "=== All ablation evals complete ==="
