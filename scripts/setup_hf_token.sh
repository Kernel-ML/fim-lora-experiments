#!/bin/bash
# Set up HuggingFace token for both terminal scripts and Jupyter notebooks.
# Run this ONCE in SageMaker Studio terminal.
#
# Usage: bash scripts/setup_hf_token.sh
# Or with token inline: HF_TOKEN=hf_xxx bash scripts/setup_hf_token.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Get token
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Enter your HuggingFace token (from https://huggingface.co/settings/tokens):"
  echo "  → Create a token with 'Read' access to gated repos (LLaMA-3 needs this)"
  read -rs HF_TOKEN
  echo ""
fi

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: No token provided."
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Persist in ~/.bashrc  (picked up by all terminal sessions + scripts)
# ---------------------------------------------------------------------------
# Remove any existing HF_TOKEN line first
sed -i '/^export HF_TOKEN=/d' ~/.bashrc 2>/dev/null || true
echo "export HF_TOKEN=${HF_TOKEN}" >> ~/.bashrc

# Also add HUGGING_FACE_HUB_TOKEN (some libs use this name)
sed -i '/^export HUGGING_FACE_HUB_TOKEN=/d' ~/.bashrc 2>/dev/null || true
echo "export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" >> ~/.bashrc

echo "✓ Token written to ~/.bashrc (for scripts)"

# ---------------------------------------------------------------------------
# 3. Write to ~/.huggingface/token  (picked up by huggingface_hub + notebooks)
# ---------------------------------------------------------------------------
mkdir -p ~/.huggingface
echo -n "${HF_TOKEN}" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
echo "✓ Token written to ~/.huggingface/token (for notebooks + huggingface_hub)"

# ---------------------------------------------------------------------------
# 4. Apply to current shell immediately
# ---------------------------------------------------------------------------
export HF_TOKEN="${HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
echo ""
echo ">>> Verifying token..."
uv run python -c "
from huggingface_hub import whoami
try:
    info = whoami()
    print(f'  Logged in as: {info[\"name\"]} ✓')
except Exception as e:
    print(f'  ERROR: {e}')
    print('  Check your token at https://huggingface.co/settings/tokens')
" 2>/dev/null || python -c "
from huggingface_hub import whoami
info = whoami()
print(f'  Logged in as: {info[\"name\"]} ✓')
"

echo ""
echo "========================================"
echo "  HF token setup complete"
echo "========================================"
echo ""
echo "For NEW terminal sessions, run:  source ~/.bashrc"
echo "Notebooks use ~/.huggingface/token automatically (no restart needed)."
