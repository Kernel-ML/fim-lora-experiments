#!/bin/bash
# Commonsense sweep: all methods × 3 seeds on LLaMA-3-8B
# Supports parallel execution across multiple GPUs (one method per GPU).
#
# Usage:
#   # Single GPU (sequential)
#   bash scripts/run_commonsense_sweep.sh
#
#   # Multi-GPU (parallel, one method per GPU — recommended on g6e.12xlarge)
#   N_GPUS=4 bash scripts/run_commonsense_sweep.sh
#
#   # Subset
#   METHODS=fim_lora SEEDS=42 bash scripts/run_commonsense_sweep.sh

set -euo pipefail

METHODS="${METHODS:-lora adalora eva fim_lora}"
SEEDS="${SEEDS:-42 1337 2024}"
RANK="${RANK:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
N_GPUS="${N_GPUS:-1}"

mkdir -p logs

echo "=== FIM-LoRA Commonsense Sweep ==="
echo "Methods: $METHODS"
echo "Seeds:   $SEEDS"
echo "Rank:    $RANK"
echo "GPUs:    $N_GPUS (parallel workers)"
echo ""

# Build job list, skipping already-completed jobs
JOBS=()
SKIPPED=0
for METHOD in $METHODS; do
  for SEED in $SEEDS; do
    RESULT="$OUTPUT_DIR/$METHOD/commonsense/r$RANK/seed$SEED/results.json"
    if [ -f "$RESULT" ]; then
      echo "  [skip] $METHOD / commonsense / r=$RANK / seed=$SEED (results.json exists)"
      SKIPPED=$((SKIPPED + 1))
    else
      JOBS+=("$METHOD|$SEED")
    fi
  done
done

TOTAL=${#JOBS[@]}
echo "Skipped: $SKIPPED (already complete)"
echo "To run:  $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
  echo "All jobs already complete."
  uv run python src/collect_results.py --output-dir "$OUTPUT_DIR" --experiment commonsense
  exit 0
fi

# Worker function: runs jobs assigned to one GPU
run_on_gpu() {
  local GPU_ID=$1
  shift
  local -a MY_JOBS=("$@")

  for JOB in "${MY_JOBS[@]}"; do
    IFS='|' read -r METHOD SEED <<< "$JOB"
    echo ">>> [GPU $GPU_ID] $METHOD / commonsense / r=$RANK / seed=$SEED"
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python src/train_commonsense.py \
      --method "$METHOD" \
      --rank "$RANK" \
      --seed "$SEED" \
      --output-dir "$OUTPUT_DIR"
    echo ""
  done
}

if [ "$N_GPUS" -eq 1 ]; then
  run_on_gpu 0 "${JOBS[@]}"
else
  # Distribute jobs round-robin across GPUs
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
      run_on_gpu "$i" "${MY_JOBS[@]}" &
      PIDS+=($!)
      echo "Started worker for GPU $i (PID ${PIDS[-1]})"
    fi
  done

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
uv run python src/collect_results.py --output-dir "$OUTPUT_DIR" --experiment commonsense
