"""Run lm-evaluation-harness on a PEFT checkpoint, respecting rank_pattern."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_peft_with_rank_pattern(base_model, checkpoint_path: str):
    """Load a PEFT checkpoint that uses rank_pattern (variable per-layer ranks).

    PEFT 0.19.1 stores rank_pattern in adapter_config.json but doesn't apply it
    during get_peft_model — all layers are allocated at r. We work around this by:
    1. Building the model at r (uniform)
    2. Loading the checkpoint weights and resizing each LoRA layer to match
    """
    import safetensors.torch
    from pathlib import Path
    import torch.nn as nn

    cfg = PeftConfig.from_pretrained(checkpoint_path)
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=cfg.task_type,
        init_lora_weights=False,
    )
    model = get_peft_model(base_model, lora_cfg)

    # Load checkpoint weights
    ckpt = Path(checkpoint_path)
    st_file = ckpt / "adapter_model.safetensors"
    bin_file = ckpt / "adapter_model.bin"
    if st_file.exists():
        ckpt_sd = safetensors.torch.load_file(str(st_file), device="cpu")
    else:
        ckpt_sd = torch.load(str(bin_file), map_location="cpu")

    # Remap lora_A.weight → lora_A.default.weight
    ckpt_sd = {
        k.replace(".lora_A.weight", ".lora_A.default.weight")
         .replace(".lora_B.weight", ".lora_B.default.weight"): v
        for k, v in ckpt_sd.items()
    }

    # Resize model LoRA layers to match checkpoint shapes, then copy weights
    model_sd = dict(model.named_parameters())
    for name, ckpt_param in ckpt_sd.items():
        if name not in model_sd:
            continue
        model_param = model_sd[name]
        if model_param.shape != ckpt_param.shape:
            # Replace the parameter tensor with correct shape, keeping device/dtype
            *parts, leaf = name.split(".")
            parent = model
            for p in parts:
                parent = getattr(parent, p)
            new_param = nn.Parameter(
                ckpt_param.to(dtype=model_param.dtype, device=model_param.device),
                requires_grad=True,
            )
            setattr(parent, leaf, new_param)
        else:
            model_param.data.copy_(ckpt_param.to(device=model_param.device))

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tasks", default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa")
    args = parser.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    cfg = PeftConfig.from_pretrained(args.checkpoint)
    if cfg.rank_pattern:
        print(f"[INFO] rank_pattern detected ({len(cfg.rank_pattern)} layers) — using manual load")
        model = load_peft_with_rank_pattern(base_model, args.checkpoint)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.checkpoint)

    model.eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    task_list = [t.strip() for t in args.tasks.split(",")]
    results = lm_eval.simple_evaluate(model=lm, tasks=task_list, log_samples=False)

    out_path = Path(args.output_dir) / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"Saved {out_path}")

    # Print summary
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or "?"
        print(f"  {task}: {acc:.4f}" if isinstance(acc, float) else f"  {task}: {acc}")


if __name__ == "__main__":
    main()
