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

    We accumulate FIM on lora_B (not lora_A) because standard LoRA initialises
    lora_B = 0, which makes ∂loss/∂lora_A = scaling × lora_B^T × upstream_grad = 0
    at initialisation — all lora_A FIM scores would be exactly zero.
    lora_A is kaiming-initialised (non-zero), so lora_B receives a meaningful
    gradient: ∂loss/∂lora_B = scaling × upstream_grad × (lora_A @ x)^T.

    Args:
        model: A PeftModel with LoRA adapters attached.
        dataloader: Iterable of dicts passed as model(**batch).
        n_batches: Number of batches to process.
        adapter_name: Active adapter name.

    Returns:
        Mapping from layer name to eFIM diagonal tensor (shape = lora_B.weight.shape).
    """
    fim_accum: dict[str, torch.Tensor] = {}
    fim_steps: dict[str, int] = defaultdict(int)

    was_training = model.training
    model.train()
    device = next(model.parameters()).device

    # Build a map: layer_name → lora_B parameter.
    # Use named_parameters() which reliably tracks the parameter objects that
    # receive .grad after backward.
    # PEFT parameter names look like:
    #   base_model.model.roberta.encoder.layer.0.attention.self.query.lora_B.default.weight
    lora_b_params: dict[str, torch.Tensor] = {}
    for param_name, param in model.named_parameters():
        if f".lora_B.{adapter_name}.weight" in param_name:
            layer_name = param_name.replace(f".lora_B.{adapter_name}.weight", "")
            param.requires_grad_(True)
            lora_b_params[layer_name] = param

    if not lora_b_params:
        warnings.warn(
            f"No lora_B parameters found for adapter '{adapter_name}'. "
            "Check that target_modules are correct and the model is a PeftModel."
        )
        return {}

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
                for layer_name, param in lora_b_params.items():
                    if param.grad is None:
                        continue
                    grad_sq = param.grad.detach() ** 2
                    if layer_name not in fim_accum:
                        fim_accum[layer_name] = torch.zeros_like(grad_sq)
                    fim_accum[layer_name].add_(grad_sq)
                    fim_steps[layer_name] += 1

            model.zero_grad()

    finally:
        model.train(was_training)

    if not fim_accum:
        warnings.warn(
            "No FIM scores accumulated — no gradients found for LoRA layers. "
            "Check: (1) dataloader produces a .loss, (2) target_modules match "
            "actual layer names, (3) model.train() is called before accumulation."
        )
        return {}

    result = {
        name: fim_accum[name] / max(fim_steps[name], 1)
        for name in fim_accum
    }

    # Warn if all scores are effectively zero (gradients didn't flow)
    all_zero = all(v.abs().max().item() < 1e-12 for v in result.values())
    if all_zero:
        warnings.warn(
            "All FIM scores are zero — gradients did not flow through LoRA layers. "
            "This usually means requires_grad was False on lora_A weights, or the "
            "loss did not depend on the LoRA parameters. Check that "
            "model.print_trainable_parameters() shows trainable params > 0."
        )

    return result


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
    Uses a two-phase water-filling algorithm:
      Phase 1 — iteratively fix layers that saturate r_max, redistributing their
                 excess budget to the remaining free layers.
      Phase 2 — apply largest-remainder rounding, then enforce r_min by stealing
                 from the lowest-importance layers above r_min.

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
    scores = {name: max(importance[name], 1e-10) for name in names}
    budget = base_r * len(names)

    result: dict[str, int] = {}
    free = list(names)
    free_budget = float(budget)

    # Phase 1: iteratively fix layers that saturate r_max, redistributing their
    # excess budget to the remaining free layers.
    while free:
        total_score = sum(scores[n] for n in free)
        raw = {n: scores[n] / total_score * free_budget for n in free}
        over_max = [n for n in free if math.floor(raw[n]) >= r_max]
        if not over_max:
            break
        for n in over_max:
            result[n] = r_max
        free_budget -= len(over_max) * r_max
        free = [n for n in free if n not in result]

    # Phase 2: LRM over remaining free layers, then enforce r_min.
    if free:
        total_score = sum(scores[n] for n in free)
        raw = {n: scores[n] / total_score * free_budget for n in free}
        floors = {n: math.floor(raw[n]) for n in free}
        rems = sorted(free, key=lambda n: raw[n] - floors[n], reverse=True)
        leftover = int(free_budget) - sum(floors.values())
        for n in rems[:leftover]:
            floors[n] += 1
        result.update(floors)

        # Enforce r_min: bump sub-minimum layers and steal from least-important donors.
        deficit = sum(max(0, r_min - result[n]) for n in free)
        for n in free:
            result[n] = max(r_min, result[n])
        if deficit:
            donors = sorted(
                [n for n in free if result[n] > r_min],
                key=lambda n: scores[n],
            )
            for donor in donors:
                give = min(result[donor] - r_min, deficit)
                result[donor] -= give
                deficit -= give
                if deficit == 0:
                    break

    return result


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
