#!/bin/bash

# Usage:
#   bash launch_eval_gsm8k_rewards.sh /path/to/checkpoint [tokenizer_name_or_path] [output_dir] [num_processes]
#
# For the GSM8K launch scripts in this repo, the tokenizer should normally be the
# base model tokenizer used for training, e.g.:
#   Qwen/Qwen2.5-1.5B-Instruct

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPEN_R1_DIR="$SCRIPT_DIR/trl-GDPO/open-r1"

CHECKPOINT_PATH=${1:?checkpoint path required}
TOKENIZER_PATH=${2:-}
OUTPUT_DIR=${3:-"$SCRIPT_DIR/gsm8k_eval_results/$(basename "$CHECKPOINT_PATH")"}
NUM_PROCESSES=${4:-1}

module load anaconda/3
module load cudatoolkit/12.4.0

conda activate gdpo-trl

cd "$OPEN_R1_DIR"

ARGS=(
    --model_name_or_path "$CHECKPOINT_PATH"
    --split test
    --output_dir "$OUTPUT_DIR"
    --per_device_eval_batch_size 8
    --max_prompt_length 512
    --max_completion_length 1024
)

if [[ -n "$TOKENIZER_PATH" ]]; then
    ARGS+=(--tokenizer_name_or_path "$TOKENIZER_PATH")
fi

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_LOG_LEVEL=info ACCELERATE_USE_DEEPSPEED=false \
    accelerate launch --num_processes "$NUM_PROCESSES" \
    src/open_r1/eval_gsm8k_rewards.py "${ARGS[@]}"
