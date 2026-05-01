"""Train and evaluate a single GLUE task with a given method.

Usage:
    uv run python src/train_glue.py \
        --method fim_lora \
        --task mnli \
        --rank 8 \
        --seed 42

Methods: lora | adalora | eva | fim_lora | random_rank
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import get_peft_model

import evaluate as hf_evaluate

from baselines import ExperimentConfig, get_adalora_config, get_eva_config, get_fim_lora_config, get_lora_config, get_random_rank_config
from fim_allocator import apply_fim_ranks

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

GLUE_TASKS = {
    "mnli":  {"dataset": "mnli",   "split_eval": "validation_matched", "metric": "accuracy", "num_labels": 3},
    "sst2":  {"dataset": "sst2",   "split_eval": "validation",          "metric": "accuracy", "num_labels": 2},
    "cola":  {"dataset": "cola",   "split_eval": "validation",          "metric": "matthews_correlation", "num_labels": 2},
    "mrpc":  {"dataset": "mrpc",   "split_eval": "validation",          "metric": "f1",       "num_labels": 2},
    "qqp":   {"dataset": "qqp",    "split_eval": "validation",          "metric": "f1",       "num_labels": 2},
    "qnli":  {"dataset": "qnli",   "split_eval": "validation",          "metric": "accuracy", "num_labels": 2},
    "stsb":  {"dataset": "stsb",   "split_eval": "validation",          "metric": "pearson",  "num_labels": 1},
    "rte":   {"dataset": "rte",    "split_eval": "validation",          "metric": "accuracy", "num_labels": 2},
}

DEBERTA_TARGET_MODULES = ["query_proj", "value_proj", "key_proj"]
MODEL_NAME = "microsoft/deberta-v3-base"

TASK_EPOCHS = {
    "mnli": 10, "qqp": 10, "qnli": 10,
    "sst2": 10, "stsb": 20, "mrpc": 30,
    "cola": 30, "rte": 30,
}


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

SENTENCE_KEYS = {
    "mnli":  ("premise", "hypothesis"),
    "sst2":  ("sentence", None),
    "cola":  ("sentence", None),
    "mrpc":  ("sentence1", "sentence2"),
    "qqp":   ("question1", "question2"),
    "qnli":  ("question", "sentence"),
    "stsb":  ("sentence1", "sentence2"),
    "rte":   ("sentence1", "sentence2"),
}


def preprocess_glue(task: str, tokenizer, max_length: int = 512):
    keys = SENTENCE_KEYS[task]

    def tokenize(examples):
        if keys[1]:
            return tokenizer(examples[keys[0]], examples[keys[1]],
                             truncation=True, max_length=max_length)
        return tokenizer(examples[keys[0]], truncation=True, max_length=max_length)

    return tokenize


def load_glue_dataset(task: str, tokenizer, max_length: int = 512):
    raw = load_dataset("glue", task)
    tokenize_fn = preprocess_glue(task, tokenizer, max_length)
    cols_to_remove = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(task: str):
    metric_name = GLUE_TASKS[task]["metric"]
    glue_metric = hf_evaluate.load("glue", task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "stsb":
            preds = logits.squeeze()
        else:
            preds = logits.argmax(-1)
        return glue_metric.compute(predictions=preds, references=labels)

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(args):
    set_seed(args.seed)
    task_cfg = GLUE_TASKS[args.task]

    cfg = ExperimentConfig(
        model_name=MODEL_NAME,
        task=args.task,
        base_r=args.rank,
        lora_alpha=args.rank * 2,       # 2r — validated stable in FP32
        lora_dropout=0.1,
        target_modules=DEBERTA_TARGET_MODULES,
        learning_rate=2e-4,
        num_epochs=args.num_epochs if args.num_epochs else TASK_EPOCHS.get(args.task, 10),
        batch_size=8,
        warmup_ratio=0.06,
        weight_decay=0.01,
        max_seq_length=512,
        fim_n_batches=args.fim_batches,
        seed=args.seed,
        gradient_accumulation_steps=2,
    )

    output_dir = Path(args.output_dir) / args.method / args.task / f"r{args.rank}" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B run
    run_name = f"{args.method}_{args.task}_r{args.rank}_seed{args.seed}"
    if not args.no_wandb:
        wandb.init(project="fim-lora", name=run_name, config=vars(args))

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_labels = task_cfg["num_labels"]
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True,
        use_safetensors=True, torch_dtype=torch.float32,
    )

    # Dataset
    dataset = load_glue_dataset(args.task, tokenizer, cfg.max_seq_length)
    train_dataset = dataset["train"]
    eval_dataset = dataset[task_cfg["split_eval"]]
    collator = DataCollatorWithPadding(tokenizer)

    # Set dataset size so cfg.total_steps() works (needed by AdaLoRA)
    cfg.train_dataset_size = len(train_dataset)
    total_steps = cfg.total_steps()

    # PEFT config
    method_map = {
        "lora":        get_lora_config,
        "adalora":     get_adalora_config,
        "eva":         get_eva_config,
        "fim_lora":    get_fim_lora_config,
        "random_rank": get_random_rank_config,
    }
    peft_config = method_map[args.method](cfg)
    model = get_peft_model(base_model, peft_config)

    # FIM rank allocation (fim_lora and random_rank only)
    if args.method == "fim_lora":
        from torch.utils.data import DataLoader
        calib_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, collate_fn=collator)
        rank_pattern = apply_fim_ranks(
            model, calib_loader, n_batches=cfg.fim_n_batches,
            r_min=args.fim_r_min, r_max=cfg.base_r * 2, verbose=True,
        )
        if not args.no_wandb:
            wandb.log({"rank_pattern": {k: v for k, v in rank_pattern.items()}})

    elif args.method == "random_rank":
        import random as _random
        from peft.tuners.lora.layer import Linear as LoraLinear
        from fim_allocator import resize_lora_layer

        _random.seed(args.seed)
        layers = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, LoraLinear) and "default" in m.lora_A]
        budget = cfg.base_r * len(layers)
        # Draw random ranks summing to budget
        rand_ranks = [max(1, min(cfg.base_r * 2, int(_random.gauss(cfg.base_r, cfg.base_r * 0.5))))
                      for _ in layers]
        # Rescale to hit budget exactly
        scale = budget / sum(rand_ranks)
        rand_ranks = [max(1, min(cfg.base_r * 2, round(r * scale))) for r in rand_ranks]
        for (name, module), r in zip(layers, rand_ranks):
            resize_lora_layer(module, "default", r, adjust_scaling=True)

    model.print_trainable_parameters()

    # Training
    # Note on batch size: DeBERTa-v3 requires FP32 (bf16/fp16 causes instability).
    # FP32 + batch=32 OOMs on L4 24GB. We use batch=16 + grad_accum=2 to match
    # AdaLoRA's effective batch size of 32 with identical total gradient steps.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,       # 16
        per_device_eval_batch_size=cfg.batch_size * 2,    # 32 (eval is inference-only, fits)
        gradient_accumulation_steps=2,                    # effective batch = 16 × 2 = 32
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=math.ceil(total_steps * cfg.warmup_ratio),
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=task_cfg["metric"],
        report_to="wandb" if not args.no_wandb else "none",
        run_name=run_name,
        seed=args.seed,
        fp16=False,    # DeBERTa-v3 requires FP32 — fp16 causes loss spikes
        bf16=False,
        dataloader_num_workers=4,
        logging_steps=50,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(args.task),
    )

    trainer.train()

    # Save results
    eval_results = trainer.evaluate()
    result = {
        "method": args.method,
        "task": args.task,
        "rank": args.rank,
        "seed": args.seed,
        "eval_results": eval_results,
        "primary_metric": task_cfg["metric"],
        "primary_score": eval_results.get(f"eval_{task_cfg['metric']}", None),
    }
    result_path = output_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[DONE] {run_name}")
    print(f"  {task_cfg['metric']}: {result['primary_score']:.4f}")
    print(f"  Saved to {result_path}")

    if not args.no_wandb:
        wandb.log({f"final/{task_cfg['metric']}": result["primary_score"]})
        wandb.finish()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "adalora", "eva", "fim_lora", "random_rank"],
                        required=True)
    parser.add_argument("--task", choices=list(GLUE_TASKS.keys()), required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fim-batches", type=int, default=8)
    parser.add_argument("--fim-r-min", type=int, default=8,
                        help="Minimum rank for FIM-LoRA allocation (ablation best: r_min=8)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    run(args)
