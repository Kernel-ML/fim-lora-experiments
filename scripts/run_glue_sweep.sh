#!/bin/bash
# Full GLUE sweep: all methods × all tasks × all ranks × 3 seeds
# Estimated: ~25 A100-hours
#
# Usage on a GPU node:
#   bash scripts/run_glue_sweep.sh
#
# To run a single cell:
#   METHOD=fim_lora TASK=mnli RANK=8 SEED=42 bash scripts/run_glue_sweep.sh

set -euo pipefail

METHODS="${METHODS:-lora adalora eva fim_lora random_rank}"
TASKS="${TASKS:-mnli sst2 cola mrpc qqp qnli stsb rte}"
RANKS="${RANKS:-2 4 8 16}"
SEEDS="${SEEDS:-42 1337 2024}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

echo "=== FIM-LoRA GLUE Sweep ==="
echo "Methods: $METHODS"
echo "Tasks:   $TASKS"
echo "Ranks:   $RANKS"
echo "Seeds:   $SEEDS"
echo ""

for METHOD in $METHODS; do
  for TASK in $TASKS; do
    for RANK in $RANKS; do
      for SEED in $SEEDS; do
        echo ">>> $METHOD / $TASK / r=$RANK / seed=$SEED"
        uv run python src/train_glue.py \
          --method "$METHOD" \
          --task "$TASK" \
          --rank "$RANK" \
          --seed "$SEED" \
          --output-dir "$OUTPUT_DIR"
        echo ""
      done
    done
  done
done

echo "=== Sweep complete. Collecting results... ==="
uv run python src/collect_results.py --output-dir "$OUTPUT_DIR" --experiment glue
