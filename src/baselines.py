"""Baseline method configurations for FIM-LoRA comparison.

All methods receive the same parameter budget. The budget is defined as:
  total_params = r × in_features × out_features, summed across target modules.

For a fair comparison at rank r:
  - LoRA:      all layers at rank r
  - AdaLoRA:   target_r=r, init_r=r*2 (standard AdaLoRA default)
  - EVA:       rank r with EVA initialization
  - FIM-LoRA:  budget = n_layers × r, allocated by FIM importance
  - Random:    budget = n_layers × r, allocated randomly (ablation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Shared configuration for all methods in one experiment run."""
    model_name: str
    task: str
    base_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    learning_rate: float
    num_epochs: int
    batch_size: int
    warmup_ratio: float
    weight_decay: float
    max_seq_length: int
    fim_n_batches: int = 8
    seed: int = 42
    output_dir: str = "results"
    wandb_project: str = "fim-lora"
    train_dataset_size: Optional[int] = None
    gradient_accumulation_steps: int = 1

    def total_steps(self) -> Optional[int]:
        if self.train_dataset_size is None:
            return None
        steps_per_epoch = self.train_dataset_size // (self.batch_size * self.gradient_accumulation_steps)
        return steps_per_epoch * self.num_epochs


def get_lora_config(cfg: ExperimentConfig):
    """Standard LoRA — fixed rank across all layers."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.base_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=_infer_task_type(cfg.task),
    )


def get_adalora_config(cfg: ExperimentConfig, total_steps: int = 0):
    """AdaLoRA — SVD-based adaptive rank allocation during training."""
    from peft import AdaLoraConfig, TaskType

    total_step = cfg.total_steps()
    # AdaLoRA schedule: tinit=warmup steps, tfinal=final freeze steps.
    # Constraint: tinit < total_step - tfinal  (budgeting phase must exist).
    # Original 10% tinit caused a loss spike to 12+ at step 50 because SVD
    # rank scores are noise before the model has warmed up. Using 20% warmup.
    # tfinal=10% leaves a final freeze phase of 10% of total steps.
    tinit = int(total_step * 0.20) if total_step else 100
    tfinal = int(total_step * 0.10) if total_step else 50

    return AdaLoraConfig(
        init_r=cfg.base_r + max(4, cfg.base_r // 4),  # modest 25% headroom
        target_r=cfg.base_r,            # end at same budget as LoRA
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=_infer_task_type(cfg.task),
        total_step=total_step,
        beta1=0.85,
        beta2=0.85,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=10,
    )


def get_eva_config(cfg: ExperimentConfig):
    """EVA — activation-variance SVD initialization."""
    from peft import LoraConfig
    from peft.tuners.lora.config import EvaConfig

    return LoraConfig(
        r=cfg.base_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=_infer_task_type(cfg.task),
        init_lora_weights="eva",
        eva_config=EvaConfig(rho=2.0, adjust_scaling_factors=True),
    )


def get_fim_lora_config(cfg: ExperimentConfig):
    """FIM-LoRA — eFIM diagonal rank allocation (our method).

    Returns a standard LoraConfig with init_lora_weights=True.
    Rank reallocation is applied separately via apply_fim_ranks().
    """
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.base_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=_infer_task_type(cfg.task),
        init_lora_weights=True,
    )


def get_random_rank_config(cfg: ExperimentConfig):
    """Random rank allocation ablation — same budget as FIM-LoRA, random assignment."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.base_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=_infer_task_type(cfg.task),
        init_lora_weights=True,
    )


def _infer_task_type(task: str):
    from peft import TaskType

    if task in {"mnli", "sst2", "cola", "mrpc", "qqp", "qnli", "stsb", "rte", "wnli",
                "boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "obqa"}:
        return TaskType.SEQ_CLS
    if task in {"squad", "squad_v2"}:
        return TaskType.QUESTION_ANS
    return TaskType.CAUSAL_LM
