# FIM-LoRA Research Plan

**Paper:** FIM-LoRA: Fisher Information Matrix-Guided Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning  
**Author:** Ramakrishnan Sathyavageeswaran  
**Target venues:** NeurIPS 2026 (May deadline) · ICLR 2027 (October deadline)  
**Repo:** https://github.com/Kernel-ML/fim-lora-experiments  
**PEFT PR:** https://github.com/huggingface/peft/pull/3204

---

## 1. Motivation

LoRA uses a fixed rank `r` across all adapter matrices. Different layers have different sensitivity to the fine-tuning loss — wasting capacity on insensitive layers while under-allocating to critical ones.

**The gap:** No existing work uses the diagonal of the empirical Fisher Information Matrix (eFIM) as the rank-allocation signal for LoRA. Existing methods use structural signals:

| Method | Rank Signal | Timing |
|--------|------------|--------|
| AdaLoRA (ICLR 2023) | SVD singular value × gradient sensitivity | During training |
| EVA (NeurIPS 2025) | Activation variance (SVD of inputs) | Pre-training calibration |
| GoRA (2025) | Gradient nuclear norm of weight matrix | Pre-training |
| ALoRA (NAACL 2024) | Per-rank gradient magnitude | During training |
| **FIM-LoRA (ours)** | **eFIM diagonal — mean squared gradient** | **Pre-training calibration** |

**Core algorithm:**

```
F_ii ≈ (1/T) Σ (∂ℓ_t / ∂θ_i)²        # eFIM diagonal
score_i = mean(F_i)                     # per-layer importance
rank_i  ∝ score_i / Σ score_j × budget  # proportional, budget = n_layers × r
rank_i  = clamp(rank_i, r_min, r_max)  # integer via largest-remainder
```

---

## 2. Datasets

### Experiment 1 — GLUE (NLU, Table 1)
- **Model:** `microsoft/deberta-v3-base` (183M params)
- **Tasks:** MNLI, SST-2, CoLA, MRPC, QQP, QNLI, STS-B, RTE
- **Metrics:** Task-specific (accuracy, F1, Matthews, Pearson) + aggregate avg
- **Rank sweep:** r ∈ {2, 4, 8, 16}
- **Seeds:** 42, 1337, 2024 (report mean ± std)
- **Why:** Exact setup from AdaLoRA paper — enables direct comparison

### Experiment 2 — Commonsense Reasoning (LLM, Table 2)
- **Model:** `meta-llama/Meta-Llama-3-8B`
- **Fine-tuning data:** commonsense_170k (standard PEFT benchmark dataset)
- **Evaluation (zero-shot):** BoolQ, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OBQA
- **Metric:** Accuracy per task + average
- **Fixed rank:** r=16 (standard for 7-8B models)
- **Why:** Current standard for LLM LoRA papers; shows generalization beyond encoders

---

## 3. Baselines

All methods use identical hyperparameters except the rank allocation mechanism. Same total trainable parameter budget.

| Method | Config | Notes |
|--------|--------|-------|
| `lora` | Fixed r across all layers | Lower bound |
| `adalora` | AdaLoraConfig, init_r=2r, target_r=r | ICLR 2023 strongest prior |
| `eva` | init_lora_weights="eva", EvaConfig(rho=2.0) | NeurIPS 2025 |
| `fim_lora` | Standard LoraConfig + apply_fim_ranks() | **Our method** |
| `random_rank` | Same budget, randomly assigned ranks | Ablation: proves FIM signal is informative |
| Full fine-tuning | All parameters | Upper bound (where feasible) |

---

## 4. Hyperparameters

### GLUE / DeBERTa
| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Warmup ratio | 6% |
| Epochs | 30 (small tasks: CoLA, MRPC, RTE) · 10 (large: MNLI, QQP) |
| Batch size | 16 per device + grad_accum=2 → effective 32 (batch=32 OOMs in FP32 on L4 24GB; step count kept identical to AdaLoRA) |
| Precision | FP32 (deberta-v3-base safetensors stores weights in FP16; forced to FP32 to prevent classifier overflow on first optimizer step) |
| Target modules | query_proj, value_proj, key_proj |
| lora_alpha | 2 × r (validated stable in FP32) |
| lora_dropout | 0.1 |
| FIM calibration | 8 batches from train set |

### LLaMA-3-8B
| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Warmup ratio | 3% |
| LR schedule | Cosine |
| Epochs | 3 |
| Batch size | 16 (grad accum 4 → effective 64) |
| Precision | bf16 |
| Target modules | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| FIM calibration | 8 batches |

### Validated Hyperparameter Notes (from empirical testing on L4 24GB)

| Param | Plan | Tested | Verdict |
|-------|------|--------|---------|
| lr | 2e-4 | ✅ stable in FP32 at steps 0–1 | Keep 2e-4 |
| lora_alpha | 2r | ✅ stable in FP32 | Keep 2r |
| batch_size | 32 | ❌ OOMs in FP32 on L4 | Changed to 16 + grad_accum=2 |
| precision | unspecified | ❌ fp16 causes classifier overflow | FP32 required for DeBERTa-v3 |

> **Any sweep run with lr=5e-5 or alpha=r must be discarded and rerun** — those are wrong relative to AdaLoRA's setup and would make the paper comparison invalid.

---

## 5. Ablation Studies

All ablations on DeBERTa/MNLI at r=8 (fastest task, most reliable signal).

| Ablation | Variable | Values | Question |
|----------|----------|--------|---------|
| A | Calibration batches | 1, 2, 4, 8, 16, 32 | How many batches are enough? |
| B | FIM aggregation | mean, max, L2-norm | Does aggregation choice matter? |
| C | FIM vs Random | — | Is the FIM signal informative vs random? |
| D | FIM + EVA init | — | Do FIM rank allocation + EVA weight init stack? |
| E | Layer analysis | by depth, by type | Which layers get more rank? (interpretability figure) |

---

## 6. Compute Plan

### SageMaker Instances
| Experiment | Instance | GPU | VRAM | Spot $/hr | Est. hours | Est. cost |
|------------|----------|-----|------|-----------|-----------|----------|
| GLUE sweep | `ml.g6.xlarge` | 1× L4 | 24 GB | ~$0.34 | ~35 hr | ~$12 |
| LLaMA-3-8B | `ml.g6e.2xlarge` | 1× L40S | 48 GB | ~$0.84 | ~40 hr | ~$34 |
| Ablations | `ml.g6.xlarge` | 1× L4 | 24 GB | ~$0.34 | ~15 hr | ~$5 |
| **Total** | | | | | | **~$51** |

### Launch Commands
```bash
# Set env vars (SageMaker Studio terminal)
export SAGEMAKER_ROLE=$(python -c "import sagemaker; print(sagemaker.get_execution_role())")
export SAGEMAKER_BUCKET=$(python -c "import sagemaker; print(sagemaker.Session().default_bucket())")
export HF_TOKEN="hf_..."

# Experiment 1 — full GLUE sweep (all jobs in parallel)
uv run python scripts/sagemaker_train.py --sweep glue --instance ml.g6.xlarge

# Experiment 2 — LLaMA-3-8B commonsense
uv run python scripts/sagemaker_train.py --sweep llama --instance ml.g6e.2xlarge

# Collect results when done
uv run python scripts/sagemaker_train.py --collect
uv run python src/collect_results.py --experiment glue
```

---

## 7. Paper Structure

**8 pages + references (ICLR format)**

| Section | Content | Length |
|---------|---------|--------|
| Introduction | Fixed-rank problem, eFIM motivation, 3 contributions | 1 page |
| Background | LoRA formulation, eFIM definition, related work table | 0.75 page |
| Method | Algorithm 1 (FIM-LoRA), complexity analysis | 1 page |
| Experiments | Table 1 (GLUE), Table 2 (LLaMA), Figure 1 (rank budget curve) | 3 pages |
| Analysis | Calibration ablation, layer heatmap, FIM+EVA combination | 0.75 page |
| Conclusion | Summary, limitations, future work | 0.5 page |

### Key Figures
1. **Figure 1 (main):** Performance vs total trainable params — FIM-LoRA Pareto curve vs baselines
2. **Figure 2:** Calibration batch ablation — accuracy vs n_batches (shows saturation at 8)
3. **Figure 3:** Layer rank heatmap — layer depth × rank assigned (interpretability)
4. **Figure 4:** FIM+EVA combination bar chart

### Citations
- LoRA (Hu et al., 2022)
- AdaLoRA (Zhang et al., ICLR 2023)
- EVA (NeurIPS 2025)
- Optimal Brain Damage (LeCun et al., NeurIPS 1990)

---

## 8. Repository Structure

```
fim-lora-experiments/
├── src/
│   ├── fim_allocator.py      # Core eFIM accumulation + rank allocation
│   ├── baselines.py          # LoRA / AdaLoRA / EVA / Random configs
│   ├── train_glue.py         # GLUE training on DeBERTaV3-base
│   ├── train_commonsense.py  # LLaMA-3-8B commonsense training
│   └── collect_results.py   # Aggregate results → CSV + tables
├── scripts/
│   ├── setup_sagemaker.sh   # One-shot env setup for Studio
│   ├── setup_hf_token.sh    # HF token persistence
│   ├── sagemaker_train.py   # SageMaker job launcher
│   ├── run_glue_sweep.sh    # Local sweep runner
│   └── eval_commonsense.sh  # lm-evaluation-harness eval
├── notebooks/
│   └── 01_smoke_test.ipynb  # End-to-end pipeline validation
├── configs/                  # YAML overrides
├── results/                  # Output (gitignored)
└── RESEARCH_PLAN.md         # This file
```

---

## 9. Current Status

| Component | Status |
|-----------|--------|
| FIM allocator (`fim_allocator.py`) | ✅ Implemented, debugging gradient flow |
| Baselines (`baselines.py`) | ✅ LoRA, AdaLoRA, EVA, Random configured |
| GLUE training script | ✅ Ready |
| LLaMA-3 training script | ✅ Ready |
| SageMaker launcher | ✅ Ready |
| Smoke test notebook | 🔧 Fixing gradient flow (named_parameters fix) |
| GLUE experiments | ⏳ Pending smoke test pass |
| LLaMA-3 experiments | ⏳ Pending |
| Ablations | ⏳ Pending |
| Paper writing | ⏳ Pending results |
| PEFT PR update | ⏳ Pending results |
