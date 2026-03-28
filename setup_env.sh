#!/bin/bash
# Setup script for the trl-GDPO conda environment
# Tested on Mila cluster with A100 GPUs
#
# PREREQUISITES:
#   - Must be run on a node with a GPU (e.g. via salloc/srun)
#   - Takes ~15 min (flash-attn compilation is slow)
#
# USAGE:
#   bash setup_env.sh
#
# AFTER SETUP, activate with:
#   module load cuda/12.4.1
#   conda activate gdpo-trl

set -euo pipefail

# 1. Load CUDA
module load cuda/12.4.1

# 2. Create conda env
conda create -n gdpo-trl python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate gdpo-trl

# 3. Install cuDNN (needed by flash-attn at runtime)
conda install -c conda-forge cudnn=9 -y

# 4. Install vllm (brings torch 2.6.0+cu124 and many deps)
pip install vllm==0.8.5.post1

# 5. Install flash-attn (compiles from source — requires GPU node)
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# 6. Install open-r1 (pins transformers==4.52.3, deepspeed, etc.)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -e "${REPO_DIR}/trl-GDPO/open-r1[dev]"

# 7. Install trl-GDPO (trl 0.18.0 with GDPO modifications)
pip install -e "${REPO_DIR}/trl-GDPO/trl-0.18.0-gdpo"

echo ""
echo "Done! Activate with:"
echo "  module load cuda/12.4.1"
echo "  conda activate gdpo-trl"
