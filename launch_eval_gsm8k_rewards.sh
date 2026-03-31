#!/bin/bash

# Usage:
#   bash launch_eval_gsm8k_rewards.sh /path/to/checkpoint [/path/to/tokenizer_or_parent_dir]

set -euo pipefail

CHECKPOINT_PATH=${1:?checkpoint path required}
TOKENIZER_PATH=${2:-}

module load anaconda/3
module load cudatoolkit/12.4.0

conda activate gdpo-trl

cd /home/mila/g/girgisro/git_repos/GDPO/trl-GDPO/open-r1/

ARGS=(
    --model_name_or_path "$CHECKPOINT_PATH"
    --split test
    --output_dir "/home/mila/g/girgisro/git_repos/GDPO/gsm8k_eval_results/$(basename "$CHECKPOINT_PATH")"
    --per_device_eval_batch_size 8
    --max_prompt_length 512
    --max_completion_length 1024
    --torch_dtype bfloat16
)

if [[ -n "$TOKENIZER_PATH" ]]; then
    ARGS+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_LOG_LEVEL=info ACCELERATE_USE_DEEPSPEED=false \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/eval_gsm8k_rewards.py "${ARGS[@]}"
