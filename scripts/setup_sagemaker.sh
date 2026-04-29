#!/bin/bash
# Setup script for SageMaker Studio JupyterLab environment
# Handles disk space issues with CUDA packages and registers Jupyter kernel
#
# Usage: bash scripts/setup_sagemaker.sh
#
# Run this ONCE after cloning the repo in SageMaker Studio terminal.

set -euo pipefail

echo "=== FIM-LoRA SageMaker Setup ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Check available disk space
# ---------------------------------------------------------------------------
echo ">>> Disk space check"
df -h /home/sagemaker-user /tmp 2>/dev/null || df -h /
echo ""

# ---------------------------------------------------------------------------
# 2. Fix: point uv cache to /tmp which has more space than home dir
#    SageMaker Studio home dir (~/.cache) fills up fast with CUDA libs (~8GB)
# ---------------------------------------------------------------------------
echo ">>> Configuring uv cache to /tmp (avoids home dir space limit)"
export UV_CACHE_DIR=/tmp/uv-cache
export TMPDIR=/tmp
mkdir -p /tmp/uv-cache

# Persist across terminal sessions
grep -q "UV_CACHE_DIR" ~/.bashrc 2>/dev/null || \
  echo 'export UV_CACHE_DIR=/tmp/uv-cache' >> ~/.bashrc

# ---------------------------------------------------------------------------
# 3. Install uv if not present
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo ">>> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc
  source ~/.cargo/env 2>/dev/null || true
else
  echo ">>> uv already installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 4. Install torch with CUDA — use index-url to get the right wheel
#    and avoid downloading all nvidia sub-packages unnecessarily
# ---------------------------------------------------------------------------
echo ""
echo ">>> Installing project dependencies (torch + CUDA via PyTorch index)..."

# SageMaker Studio instances already have CUDA drivers — use pre-built torch
# wheels from PyTorch's own index to avoid the nvidia-* sub-package bloat
uv sync --no-install-package torch 2>/dev/null || true

# Install torch separately from PyTorch index (smaller, no nvidia-* extras)
uv pip install torch \
  --index-url https://download.pytorch.org/whl/cu121 \
  --no-deps \
  2>&1 | tail -5

# Install remaining deps
uv sync --no-build-isolation 2>&1 | tail -5

echo ""
echo ">>> Verifying torch + CUDA"
uv run python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.version.cuda}')
print(f'  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')
print(f'  Available:  {torch.cuda.is_available()}')
"

# ---------------------------------------------------------------------------
# 5. Register as a Jupyter kernel so notebooks can use this env
# ---------------------------------------------------------------------------
echo ""
echo ">>> Registering Jupyter kernel: fim-lora"

uv pip install ipykernel
uv run python -m ipykernel install \
  --user \
  --name fim-lora \
  --display-name "FIM-LoRA (uv)"

echo "  ✓ Kernel registered as 'FIM-LoRA (uv)'"
echo "  → In JupyterLab: Kernel → Change Kernel → 'FIM-LoRA (uv)'"

# ---------------------------------------------------------------------------
# 6. Verify key imports
# ---------------------------------------------------------------------------
echo ""
echo ">>> Verifying key imports"
uv run python -c "
import torch, transformers, peft, datasets, evaluate, wandb
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  datasets:       {datasets.__version__}')
print(f'  evaluate:       {evaluate.__version__}')
print(f'  wandb:          {wandb.__version__}')
print()
print('  All imports OK ✓')
"

# ---------------------------------------------------------------------------
# 7. Quick FIM allocator smoke check (CPU, no GPU needed)
# ---------------------------------------------------------------------------
echo ""
echo ">>> FIM allocator import check"
uv run python -c "
import sys; sys.path.insert(0, 'src')
from fim_allocator import accumulate_fim, compute_importance, allocate_ranks, apply_fim_ranks
print('  fim_allocator imports OK ✓')
from baselines import get_lora_config, get_adalora_config, get_eva_config, get_fim_lora_config
print('  baselines imports OK ✓')
"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Setup complete ✓"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Refresh JupyterLab (F5) to see 'FIM-LoRA (uv)' kernel"
echo "  2. Open notebooks/01_smoke_test.ipynb → select 'FIM-LoRA (uv)' kernel"
echo "  3. Run smoke test to validate pipeline"
echo "  4. Launch experiments:"
echo ""
echo "     # Single job (validate before full sweep)"
echo "     uv run python src/train_glue.py --method fim_lora --task rte --rank 4 --no-wandb"
echo ""
echo "     # Full GLUE sweep"
echo "     bash scripts/run_glue_sweep.sh"
echo ""
echo "Disk usage after setup:"
df -h /home/sagemaker-user /tmp 2>/dev/null | head -5 || df -h / | head -3
