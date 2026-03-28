#!/bin/bash
# Setup script for the trl-GDPO environment on Compute Canada (Alliance)
#
# Two-phase install because compute nodes have no internet:
#   Phase 1 (login node):  download & install all pure-Python / prebuilt wheels
#   Phase 2 (compute node): compile flash-attn (needs GPU + CUDA toolkit)
#
# USAGE:
#   # On a login node:
#   bash setup_env_cc.sh login
#
#   # Then on a compute node (salloc --gres=gpu:1 ...):
#   bash setup_env_cc.sh compute
#
# AFTER SETUP, activate with:
#   module load python/3.10 cuda/12.4 cudnn/9
#   source $HOME/envs/gdpo-trl/bin/activate

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$HOME/envs/gdpo-trl"
FLASH_ATTN_VERSION="2.7.4.post1"

phase_login() {
    echo "=== Phase 1: Login node (downloading & installing packages) ==="

    # Load modules
    module load python/3.10 cuda/12.4 cudnn/9

    # Install uv if not available
    if ! command -v uv &> /dev/null; then
        pip install --user uv
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Create virtual environment
    uv venv "$ENV_DIR" --python python3.10
    source "$ENV_DIR/bin/activate"

    # Install all frozen deps (except flash-attn and editable packages)
    uv pip install -r "$REPO_DIR/requirements-cc.txt"

    # Install open-r1 and trl-GDPO as editable
    uv pip install -e "$REPO_DIR/trl-GDPO/open-r1[dev]" --no-deps
    uv pip install -e "$REPO_DIR/trl-GDPO/trl-0.18.0-gdpo" --no-deps

    # Download flash-attn source so it's cached for phase 2
    uv pip download flash-attn==$FLASH_ATTN_VERSION --no-binary flash-attn -d "$REPO_DIR/.flash-attn-src"

    echo ""
    echo "Phase 1 done! Now run on a compute node:"
    echo "  salloc --gres=gpu:1 --time=1:00:00 --mem=32G"
    echo "  bash $REPO_DIR/setup_env_cc.sh compute"
}

phase_compute() {
    echo "=== Phase 2: Compute node (compiling flash-attn) ==="

    # Load modules
    module load python/3.10 cuda/12.4 cudnn/9
    source "$ENV_DIR/bin/activate"

    # Compile and install flash-attn from the pre-downloaded source
    pip install --no-index --find-links "$REPO_DIR/.flash-attn-src" \
        flash-attn==$FLASH_ATTN_VERSION --no-build-isolation --no-cache-dir

    # Verify
    python -c "
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
import transformers; print(f'transformers {transformers.__version__}')
import trl; print(f'trl {trl.__version__}')
import vllm; print(f'vllm {vllm.__version__}')
import flash_attn; print(f'flash-attn {flash_attn.__version__}')
import deepspeed; print(f'deepspeed {deepspeed.__version__}')
print('All imports OK!')
"

    echo ""
    echo "Setup complete! Activate with:"
    echo "  module load python/3.10 cuda/12.4 cudnn/9"
    echo "  source $ENV_DIR/bin/activate"
}

case "${1:-}" in
    login)   phase_login ;;
    compute) phase_compute ;;
    *)
        echo "Usage: bash setup_env_cc.sh [login|compute]"
        echo "  login   - Run on login node (downloads packages)"
        echo "  compute - Run on compute node (compiles flash-attn)"
        exit 1
        ;;
esac
