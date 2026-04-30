#!/bin/bash
# Full GLUE sweep: all methods × all tasks × all ranks × 3 seeds
# Supports parallel execution across multiple GPUs on one machine.
#
# Usage:
#   # Single GPU (sequential)
#   bash scripts/run_glue_sweep.sh
#
#   # Multi-GPU (parallel, one job per GPU)
#   N_GPUS=4 bash scripts/run_glue_sweep.sh
#
#   # Subset
#   METHODS=fim_lora TASKS=mnli RANKS=8 SEEDS=42 bash scripts/run_glue_sweep.sh

set -euo pipefail

METHODS="${METHODS:-lora adalora eva fim_lora random_rank}"
TASKS="${TASKS:-mnli sst2 cola mrpc qqp qnli stsb rte}"
RANKS="${RANKS:-2 4 8 16}"
SEEDS="${SEEDS:-42 1337 2024}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
N_GPUS="${N_GPUS:-1}"

# Tee all output to a timestamped log file
mkdir -p logs
LOG_FILE="logs/sweep_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo ""

echo "=== FIM-LoRA GLUE Sweep ==="
echo "Methods: $METHODS"
echo "Tasks:   $TASKS"
echo "Ranks:   $RANKS"
echo "Seeds:   $SEEDS"
echo "GPUs:    $N_GPUS (parallel workers)"
echo ""

# Build the full job list, skipping already-completed jobs
JOBS=()
SKIPPED=0
for METHOD in $METHODS; do
  for TASK in $TASKS; do
    for RANK in $RANKS; do
      for SEED in $SEEDS; do
        RESULT="$OUTPUT_DIR/$METHOD/$TASK/r$RANK/seed$SEED/results.json"
        if [ -f "$RESULT" ]; then
          echo "  [skip] $METHOD / $TASK / r=$RANK / seed=$SEED (results.json exists)"
          SKIPPED=$((SKIPPED + 1))
        else
          JOBS+=("$METHOD|$TASK|$RANK|$SEED")
        fi
      done
    done
  done
done

TOTAL=${#JOBS[@]}
echo "Skipped: $SKIPPED (already complete)"
echo "To run:  $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
  echo "All jobs already complete."
  uv run python src/collect_results.py --output-dir "$OUTPUT_DIR" --experiment glue
  exit 0
fi

# Worker function: runs jobs assigned to one GPU
run_on_gpu() {
  local GPU_ID=$1
  shift
  local -a MY_JOBS=("$@")

  for JOB in "${MY_JOBS[@]}"; do
    IFS='|' read -r METHOD TASK RANK SEED <<< "$JOB"
    echo ">>> [GPU $GPU_ID] $METHOD / $TASK / r=$RANK / seed=$SEED"
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python src/train_glue.py \
      --method "$METHOD" \
      --task "$TASK" \
      --rank "$RANK" \
      --seed "$SEED" \
      --output-dir "$OUTPUT_DIR"
    echo ""
  done
}

if [ "$N_GPUS" -eq 1 ]; then
  # Simple sequential path
  run_on_gpu 0 "${JOBS[@]}"
else
  # Distribute jobs round-robin across GPUs, run each GPU's queue in background
  declare -a GPU_JOBS
  for ((i=0; i<N_GPUS; i++)); do GPU_JOBS[$i]=""; done

  for ((j=0; j<TOTAL; j++)); do
    GPU=$((j % N_GPUS))
    GPU_JOBS[$GPU]+="${JOBS[$j]} "
  done

  PIDS=()
  for ((i=0; i<N_GPUS; i++)); do
    read -ra MY_JOBS <<< "${GPU_JOBS[$i]}"
    run_on_gpu "$i" "${MY_JOBS[@]}" &
    PIDS+=($!)
    echo "Started worker for GPU $i (PID ${PIDS[-1]})"
  done

  # Wait for all GPU workers, propagate first failure
  FAILED=0
  for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
  done

  if [ "$FAILED" -ne 0 ]; then
    echo "One or more GPU workers failed — check logs above."
    exit 1
  fi
fi

echo "=== Sweep complete. Collecting results... ==="
uv run python src/collect_results.py --output-dir "$OUTPUT_DIR" --experiment glue
