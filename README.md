# FIM-LoRA: Experiment Harness

Reproducible experiment code for:

> **FIM-LoRA: Fisher Information Matrix-Guided Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning**

## Method

FIM-LoRA runs a small calibration pass before training, accumulates squared gradients (eFIM diagonal) per LoRA layer, and reallocates ranks proportionally to each layer's loss sensitivity.

```
rank_i ∝ mean(F_i) / Σ mean(F_j) × budget    budget = n_layers × r
```

## Setup

```bash
uv sync
bash scripts/smoke_test.sh   # CPU smoke test (~10 min)
```

## Experiments

```bash
# Experiment 1 — GLUE sweep (~25 A100-hours)
bash scripts/run_glue_sweep.sh

# Single run
uv run python src/train_glue.py --method fim_lora --task mnli --rank 8 --seed 42

# Experiment 2 — LLaMA-3-8B commonsense
uv run python src/train_commonsense.py --method fim_lora --rank 16 --seed 42
CHECKPOINT=results/fim_lora/commonsense/r16/seed42/checkpoint-final bash scripts/eval_commonsense.sh

# Collect results
uv run python src/collect_results.py --output-dir results --experiment glue
```

## Methods

| Method | Rank Signal | Timing |
|--------|-------------|--------|
| `lora` | Fixed rank | — |
| `adalora` | SVD singular value × sensitivity | During training |
| `eva` | Activation variance (SVD) | Pre-training calibration |
| `fim_lora` | **eFIM diagonal (mean²  gradient)** | Pre-training calibration |
| `random_rank` | Random (ablation) | Pre-training |
