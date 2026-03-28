#!/bin/bash
# Run GDPO training on Compute Canada (Tamia cluster)
#
# USAGE (from a compute node with GPUs):
#   module load python/3.10 cuda/12.6 cudnn
#   source ~/envs/gdpo-trl/bin/activate
#   bash run_gsm8k_gdpo_cc.sh

set -euo pipefail

OUTPUT_DIR="${SCRATCH}/gdpo-output/Qwen2.5-1.5B-gsm8k-GDPO-3-rewards"
mkdir -p "$OUTPUT_DIR"

WANDB_MODE=offline WANDB_DIR="${SCRATCH}/gdpo-output" HF_HUB_OFFLINE=1 CUDA_HOME="$CUDA_PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3_8gpu.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/gdpo_gsm8k/config.yaml \
    --output_dir "$OUTPUT_DIR" \
    --vllm_mode colocate
