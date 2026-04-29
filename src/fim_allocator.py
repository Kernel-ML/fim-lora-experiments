"""FIM-guided rank allocation for LoRA.

Core algorithm:
  1. Run n_batches calibration forward+backward passes
  2. Accumulate squared gradients (eFIM diagonal) per LoRA layer
  3. Compute mean eFIM score per layer as importance
  4. Allocate integer ranks proportional to importance under a fixed budget
  5. Resize lora_A / lora_B and update rank_pattern in LoraConfig
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Step 1 — accumulate eFIM diagonal
# ---------------------------------------------------------------------------


def accumulate_fim(
    model: nn.Module,
    dataloader,
    n_batches: int,
    adapter_name: str = "default",
) -> dict[str, torch.Tensor]:
    """Run calibration passes and accumulate mean squared gradients per LoRA layer.

    Args:
        model: A PeftModel with LoRA adapters attached.
        dataloader: Iterable of dicts passed as model(**batch).
        n_batches: Number of batches to process.
        adapter_name: Active adapter name.

    Returns:
        Mapping from layer name to eFIM diagonal tensor (shape = lora_A.weight.shape).
    """
    from peft.tuners.lora import LoraModel
    from peft.tuners.lora.layer import Linear as LoraLinear

    fim_accum: dict[str, torch.Tensor] = {}
    fim_steps: dict[str, int] = defaultdict(int)

    was_training = model.training
    model.train()
    device = next(model.parameters()).device

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()

            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, LoraLinear) and adapter_name in module.lora_A:
                        w = module.lora_A[adapter_name].weight
                        if w.grad is None:
                            continue
                        grad_sq = w.grad.detach() ** 2
                        if name not in fim_accum:
                            fim_accum[name] = torch.zeros_like(grad_sq)
                        fim_accum[name].add_(grad_sq)
                        fim_steps[name] += 1

            model.zero_grad()

    finally:
        model.train(was_training)

    if not fim_accum:
        warnings.warn(
            "No FIM scores accumulated — no gradients found for LoRA layers. "
            "Check that your dataloader produces a loss and target_modules are correct."
        )

    return {
        name: fim_accum[name] / max(fim_steps[name], 1)
        for name in fim_accum
    }


# ---------------------------------------------------------------------------
# Step 2 — layer importance score
# ---------------------------------------------------------------------------


def compute_importance(
    fim_scores: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Aggregate eFIM diagonal per layer to a scalar importance score (mean).

    Args:
        fim_scores: Mapping from layer name to eFIM diagonal tensor.

    Returns:
        Mapping from layer name to scalar importance score.
    """
    return {name: fim.mean().item() for name, fim in fim_scores.items()}


# ---------------------------------------------------------------------------
# Step 3 — rank allocation under budget
# ---------------------------------------------------------------------------


def allocate_ranks(
    importance: dict[str, float],
    base_r: int,
    r_min: int = 1,
    r_max: Optional[int] = None,
) -> dict[str, int]:
    """Allocate integer ranks proportional to importance under a fixed budget.

    Budget = n_layers × base_r (mean rank preserved).
    Uses the largest-remainder method for integer rounding.

    Args:
        importance: Layer-name → scalar importance score.
        base_r: Original LoRA rank (defines per-layer budget).
        r_min: Minimum rank per layer.
        r_max: Maximum rank per layer. Defaults to 2 * base_r.

    Returns:
        Mapping from layer name to allocated integer rank.
    """
    if not importance:
        return {}

    r_max = r_max if r_max is not None else 2 * base_r
    names = list(importance.keys())
    scores = [max(importance[n], 1e-10) for n in names]
    total_score = sum(scores)
    budget = base_r * len(names)

    raw = [s / total_score * budget for s in scores]
    floors = [math.floor(x) for x in raw]
    remainders = sorted(enumerate(raw[i] - floors[i] for i in range(len(raw))),
                        key=lambda x: -x[1])
    remainder_budget = budget - sum(floors)
    for j in range(int(remainder_budget)):
        floors[remainders[j][0]] += 1

    return {names[i]: max(r_min, min(r_max, floors[i])) for i in range(len(names))}


# ---------------------------------------------------------------------------
# Step 4 — resize adapter weights
# ---------------------------------------------------------------------------


def resize_lora_layer(
    layer,
    adapter_name: str,
    new_r: int,
    adjust_scaling: bool = True,
) -> None:
    """Resize lora_A and lora_B to new_r in-place.

    Preserves existing weights up to min(old_r, new_r).
    Extra rows in A are kaiming-initialized; extra cols in B are zeros.

    Args:
        layer: A LoraLayer instance.
        adapter_name: Adapter to resize.
        new_r: Target rank.
        adjust_scaling: If True, rescales layer.scaling to preserve lora_alpha/r.
    """
    if adapter_name not in layer.lora_A:
        return

    lora_A = layer.lora_A[adapter_name]
    lora_B = layer.lora_B[adapter_name]
    old_r = lora_A.weight.shape[0]

    if old_r == new_r:
        return

    device, dtype = lora_A.weight.device, lora_A.weight.dtype
    in_f = lora_A.weight.shape[1]
    out_f = lora_B.weight.shape[0]
    copy_r = min(old_r, new_r)

    new_A = torch.zeros(new_r, in_f, device=device, dtype=dtype)
    new_B = torch.zeros(out_f, new_r, device=device, dtype=dtype)

    with torch.no_grad():
        new_A[:copy_r].copy_(lora_A.weight[:copy_r])
        new_B[:, :copy_r].copy_(lora_B.weight[:, :copy_r])
        if new_r > old_r:
            nn.init.kaiming_uniform_(new_A[old_r:], a=math.sqrt(5))

    lora_A.weight = nn.Parameter(new_A)
    lora_B.weight = nn.Parameter(new_B)
    layer.r[adapter_name] = new_r

    if adjust_scaling and adapter_name in layer.scaling:
        layer.scaling[adapter_name] *= old_r / new_r


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_fim_ranks(
    model: nn.Module,
    dataloader,
    n_batches: int = 8,
    r_min: int = 1,
    r_max: Optional[int] = None,
    adjust_scaling: bool = True,
    adapter_name: str = "default",
    verbose: bool = True,
) -> dict[str, int]:
    """Full FIM rank allocation pipeline.

    Runs calibration, computes importance, allocates ranks, resizes adapters.
    Call this after get_peft_model() and before training.

    Args:
        model: PeftModel with LoRA adapters.
        dataloader: Calibration dataloader (same data as fine-tuning).
        n_batches: Calibration batches. Default 8.
        r_min: Minimum rank per layer.
        r_max: Maximum rank per layer (default: 2 × base_r from config).
        adjust_scaling: Preserve lora_alpha/r scaling after reallocation.
        adapter_name: Active adapter name.
        verbose: Print rank allocation summary.

    Returns:
        Mapping from layer name to allocated rank (for logging/analysis).
    """
    from peft.tuners.lora.layer import Linear as LoraLinear

    if verbose:
        print(f"[FIM] Accumulating eFIM over {n_batches} calibration batches...")

    fim_scores = accumulate_fim(model, dataloader, n_batches, adapter_name)

    if not fim_scores:
        warnings.warn("[FIM] No scores accumulated — ranks unchanged.")
        return {}

    importance = compute_importance(fim_scores)

    # Infer base_r from the model config
    base_r = model.peft_config[adapter_name].r
    rank_pattern = allocate_ranks(importance, base_r=base_r, r_min=r_min, r_max=r_max)

    # Apply resizing
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear) and name in rank_pattern:
            resize_lora_layer(module, adapter_name, rank_pattern[name], adjust_scaling)

    # Persist rank_pattern in config for serialisation
    model.peft_config[adapter_name].rank_pattern.update(
        {k: v for k, v in rank_pattern.items() if v != base_r}
    )

    if verbose:
        ranks = sorted(rank_pattern.items(), key=lambda x: -x[1])
        print(f"[FIM] Rank allocation (base_r={base_r}, budget={base_r * len(ranks)}):")
        for name, r in ranks:
            marker = "▲" if r > base_r else ("▼" if r < base_r else "=")
            print(f"  {marker} {name:60s}  r={r}")

    return rank_pattern
