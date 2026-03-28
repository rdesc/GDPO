#!/bin/bash
# Setup script for the trl-GDPO environment on Compute Canada (Tamia cluster)
#
# Single-phase install — runs entirely on a login node.
# flash-attn is installed via a prebuilt wheel (no GPU compilation needed).
#
# USAGE:
#   bash setup_env_cc.sh
#
# AFTER SETUP, activate with:
#   module load python/3.10 cuda/12.6 cudnn
#   source $HOME/envs/gdpo-trl/bin/activate

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$HOME/envs/gdpo-trl"
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu124torch2.6-cp310-cp310-linux_x86_64.whl"

echo "=== Setting up trl-GDPO environment ==="

# Load modules (Tamia cluster: cuda/12.6 + cudnn/9.10, no 12.4)
module load python/3.10 cuda/12.6 cudnn

# Install uv if not available
if ! command -v uv &> /dev/null; then
    pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
uv venv "$ENV_DIR" --python python3.10
source "$ENV_DIR/bin/activate"

# Install all frozen deps via uv (bypasses CC pip config, pulls PyPI binary wheels)
# --extra-index-url needed for CUDA-enabled torch/torchvision/torchaudio wheels
uv pip install -r "$REPO_DIR/requirements-cc.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match

# Install flash-attn from prebuilt wheel (no compilation needed)
uv pip install "$FLASH_ATTN_WHEEL"

# Install open-r1 and trl-GDPO as editable
uv pip install -e "$REPO_DIR/trl-GDPO/open-r1[dev]" --no-deps
uv pip install -e "$REPO_DIR/trl-GDPO/trl-0.18.0-gdpo" --no-deps

# Download HF model and dataset for offline use on compute nodes
python "$REPO_DIR/download_hf_assets.py"

echo ""
echo "Setup complete! Activate with:"
echo "  module load python/3.10 cuda/12.6 cudnn"
echo "  source $ENV_DIR/bin/activate"
echo "  export CUDA_HOME=\$CUDA_PATH"
