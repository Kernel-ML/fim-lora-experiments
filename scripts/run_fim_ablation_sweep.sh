#!/bin/bash
# FIM-LoRA ablation sweep: vary r_min and n_batches on commonsense.
#
# Usage:
#   N_GPUS=8 bash scripts/run_fim_ablation_sweep.sh

set -euo pipefail

SEEDS="${SEEDS:-42 1337 2024}"
RANK="${RANK:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
N_GPUS="${N_GPUS:-1}"

mkdir -p logs

# Ablation configs: "label|r_min|n_batches"
ABLATIONS=(
  "fim_lora_rmin4|4|8"
  "fim_lora_rmin8|8|8"
  "fim_lora_nb32|1|32"
  "fim_lora_rmin4_nb32|4|32"
)

echo "=== FIM-LoRA Ablation Sweep ==="
echo "Seeds:     $SEEDS"
echo "Rank:      $RANK"
echo "GPUs:      $N_GPUS"
echo "Ablations: ${#ABLATIONS[@]}"
echo ""

# Build job list
JOBS=()
SKIPPED=0
for ABLATION in "${ABLATIONS[@]}"; do
  IFS='|' read -r LABEL R_MIN N_BATCHES <<< "$ABLATION"
  for SEED in $SEEDS; do
    RESULT="$OUTPUT_DIR/$LABEL/commonsense/r$RANK/seed$SEED/results.json"
    CKPT="$OUTPUT_DIR/$LABEL/commonsense/r$RANK/seed$SEED/checkpoint-final"
    if [ -f "$RESULT" ] || [ -d "$CKPT" ]; then
      echo "  [skip] $LABEL / seed$SEED"
      SKIPPED=$((SKIPPED + 1))
    else
      JOBS+=("$LABEL|$SEED|$R_MIN|$N_BATCHES")
    fi
  done
done

TOTAL=${#JOBS[@]}
echo "Skipped: $SKIPPED"
echo "To run:  $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
  echo "All jobs already complete."
  exit 0
fi

run_on_gpu() {
  local GPU_ID=$1
  shift
  local -a MY_JOBS=("$@")

  for JOB in "${MY_JOBS[@]}"; do
    IFS='|' read -r LABEL SEED R_MIN N_BATCHES <<< "$JOB"
    echo ">>> [GPU $GPU_ID] $LABEL / seed=$SEED (r_min=$R_MIN, n_batches=$N_BATCHES)"
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python src/train_commonsense.py \
      --method fim_lora \
      --rank "$RANK" \
      --seed "$SEED" \
      --output-dir "$OUTPUT_DIR/$LABEL" \
      --fim-r-min "$R_MIN" \
      --fim-n-batches "$N_BATCHES"
    echo ""
  done
}

if [ "$N_GPUS" -eq 1 ]; then
  run_on_gpu 0 "${JOBS[@]}"
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
    echo "One or more workers failed."
    exit 1
  fi
fi

echo "=== Ablation sweep complete ==="
