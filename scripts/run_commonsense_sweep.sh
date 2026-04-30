#!/bin/bash
# Commonsense sweep: all methods × 3 seeds on LLaMA-3-8B (single GPU, sequential)
#
# Usage:
#   bash scripts/run_commonsense_sweep.sh
#
#   # Subset
#   METHODS=fim_lora SEEDS=42 bash scripts/run_commonsense_sweep.sh

set -euo pipefail

METHODS="${METHODS:-lora adalora eva fim_lora}"
SEEDS="${SEEDS:-42 1337 2024}"
RANK="${RANK:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Tee all output to a timestamped log file
mkdir -p logs
LOG_FILE="logs/commonsense_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo ""

echo "=== FIM-LoRA Commonsense Sweep ==="
echo "Methods: $METHODS"
echo "Seeds:   $SEEDS"
echo "Rank:    $RANK"
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

# Run sequentially on single GPU
for JOB in "${JOBS[@]}"; do
  IFS='|' read -r METHOD SEED <<< "$JOB"
  echo ">>> $METHOD / commonsense / r=$RANK / seed=$SEED"
  uv run python src/train_commonsense.py \
    --method "$METHOD" \
    --rank "$RANK" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
  echo ""
done

echo "=== Sweep complete. Collecting results... ==="
uv run python src/collect_results.py --output-dir "$OUTPUT_DIR" --experiment commonsense
