#!/bin/bash
# Smoke test — runs FIM-LoRA on RTE (smallest GLUE task) for 2 epochs
# to verify the full pipeline works before renting GPU time.
# CPU-compatible (slow but functional).
#
# Usage: bash scripts/smoke_test.sh

set -euo pipefail

echo "=== FIM-LoRA Smoke Test (RTE, r=4, 2 epochs) ==="

uv run python src/train_glue.py \
  --method fim_lora \
  --task rte \
  --rank 4 \
  --seed 42 \
  --fim-batches 2 \
  --no-wandb \
  --output-dir results/smoke_test

echo ""
echo "=== LoRA baseline (same config) ==="

uv run python src/train_glue.py \
  --method lora \
  --task rte \
  --rank 4 \
  --seed 42 \
  --no-wandb \
  --output-dir results/smoke_test

echo ""
echo "=== Smoke test passed. Check results/smoke_test/ for outputs. ==="

uv run python src/collect_results.py \
  --output-dir results/smoke_test \
  --experiment glue
