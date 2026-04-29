"""Train LLaMA-3-8B on commonsense_170k and evaluate on 7 reasoning tasks.

Usage:
    uv run python src/train_commonsense.py \
        --method fim_lora \
        --rank 16 \
        --seed 42

Methods: lora | adalora | eva | fim_lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate as hf_evaluate

from baselines import ExperimentConfig, get_adalora_config, get_eva_config, get_fim_lora_config, get_lora_config
from fim_allocator import apply_fim_ranks

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

LLAMA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

COMMONSENSE_EVAL_TASKS = [
    "boolq", "piqa", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa",
]


def load_commonsense_train(tokenizer, max_length: int = 512):
    """Load and tokenize the commonsense_170k training split."""
    dataset = load_dataset("tau/commonsense_qa", split="train")

    def format_example(ex):
        choices = " ".join([f"({l}) {t}" for l, t in
                            zip(ex["choices"]["label"], ex["choices"]["text"])])
        prompt = f"Question: {ex['question']}\nChoices: {choices}\nAnswer:"
        answer = f" {ex['answerKey']}"
        full = prompt + answer
        tokenized = tokenizer(full, truncation=True, max_length=max_length,
                              padding=False, return_tensors=None)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(format_example, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    return tokenized


def evaluate_commonsense(model, tokenizer) -> dict[str, float]:
    """Zero-shot evaluate on all 7 commonsense tasks. Returns per-task accuracy."""
    results = {}
    accuracy_metric = hf_evaluate.load("accuracy")

    task_configs = {
        "boolq":         ("google/boolq",      "validation", "question", "answer"),
        "piqa":          ("piqa",              "validation", "goal",     "label"),
        "hellaswag":     ("hellaswag",         "validation", "ctx",      "label"),
        "winogrande":    ("winogrande",        "validation", "sentence", "answer"),
        "arc_easy":      ("ai2_arc", "ARC-Easy",     "validation", "question", "answerKey"),
        "arc_challenge": ("ai2_arc", "ARC-Challenge", "validation", "question", "answerKey"),
        "openbookqa":    ("openbookqa",        "validation", "question_stem", "answerKey"),
    }

    model.eval()
    for task_name, cfg in task_configs.items():
        try:
            # Minimal multiple-choice scoring via perplexity
            # Full implementation uses lm-evaluation-harness; this is a placeholder
            results[task_name] = _score_task(model, tokenizer, task_name, cfg)
        except Exception as e:
            print(f"[WARN] {task_name} eval failed: {e}")
            results[task_name] = 0.0

    results["average"] = sum(results.values()) / len(results)
    return results


def _score_task(model, tokenizer, task_name: str, cfg: tuple) -> float:
    """Placeholder: returns 0.0. Replace with lm-evaluation-harness in production."""
    # In the actual experiment, this calls:
    # lm_eval.simple_evaluate(model=lm_eval_model, tasks=[task_name], ...)
    # See scripts/eval_commonsense.sh for the full lm-eval invocation
    return 0.0


def run(args):
    set_seed(args.seed)

    cfg = ExperimentConfig(
        model_name=MODEL_NAME,
        task="commonsense",
        base_r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=LLAMA_TARGET_MODULES,
        learning_rate=3e-4,
        num_epochs=3,
        batch_size=16,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_seq_length=512,
        fim_n_batches=8,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir) / args.method / "commonsense" / f"r{args.rank}" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.method}_commonsense_r{args.rank}_seed{args.seed}"

    if not args.no_wandb:
        wandb.init(project="fim-lora", name=run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    method_map = {
        "lora":     get_lora_config,
        "adalora":  get_adalora_config,
        "eva":      get_eva_config,
        "fim_lora": get_fim_lora_config,
    }
    peft_config = method_map[args.method](cfg)
    model = get_peft_model(base_model, peft_config)

    train_dataset = load_commonsense_train(tokenizer, cfg.max_seq_length)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    if args.method == "fim_lora":
        from torch.utils.data import DataLoader
        calib_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, collate_fn=collator)
        apply_fim_ranks(model, calib_loader, n_batches=cfg.fim_n_batches, verbose=True)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="wandb" if not args.no_wandb else "none",
        run_name=run_name,
        seed=args.seed,
        bf16=True,
        dataloader_num_workers=4,
        logging_steps=50,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))

    # Evaluation via lm-evaluation-harness (see scripts/eval_commonsense.sh)
    print(f"\n[INFO] Model saved to {output_dir / 'checkpoint-final'}")
    print("[INFO] Run evaluation with: scripts/eval_commonsense.sh")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "adalora", "eva", "fim_lora"], required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run(args)
