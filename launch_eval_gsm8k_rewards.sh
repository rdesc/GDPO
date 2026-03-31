#!/bin/bash

# Usage:
#   bash launch_eval_gsm8k_rewards.sh /path/to/checkpoint [tokenizer_name_or_path] [output_dir] [num_processes] [accelerate_config] [max_eval_batches]
#
# For the GSM8K launch scripts in this repo, the tokenizer should normally be the
# base model tokenizer used for training, e.g.:
#   Qwen/Qwen2.5-1.5B-Instruct

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPEN_R1_DIR="$SCRIPT_DIR/trl-GDPO/open-r1"

CHECKPOINT_PATH=${1:?checkpoint path required}
TOKENIZER_PATH=${2:-}

CHECKPOINT_BASENAME=$(basename "$CHECKPOINT_PATH")
CHECKPOINT_PARENT_BASENAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
if [[ "$CHECKPOINT_BASENAME" == checkpoint-* ]]; then
    DEFAULT_OUTPUT_NAME="${CHECKPOINT_PARENT_BASENAME}_${CHECKPOINT_BASENAME}"
else
    DEFAULT_OUTPUT_NAME="$CHECKPOINT_BASENAME"
fi

OUTPUT_DIR=${3:-"$SCRIPT_DIR/gsm8k_eval_results/$DEFAULT_OUTPUT_NAME"}
NUM_PROCESSES=${4:-1}
ACCELERATE_CONFIG=${5:-}
MAX_EVAL_BATCHES=${6:-}

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
if [[ -n "$MAX_EVAL_BATCHES" ]]; then
    ARGS+=(--max_eval_batches "$MAX_EVAL_BATCHES")
fi

if [[ -z "$ACCELERATE_CONFIG" && "$NUM_PROCESSES" == "8" && -f "recipes/accelerate_configs/zero3_8gpu.yaml" ]]; then
    ACCELERATE_CONFIG="recipes/accelerate_configs/zero3_8gpu.yaml"
fi

if [[ -n "$ACCELERATE_CONFIG" ]]; then
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_LOG_LEVEL=info \
        accelerate launch --config_file "$ACCELERATE_CONFIG" \
        src/open_r1/eval_gsm8k_rewards.py "${ARGS[@]}"
else
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_LOG_LEVEL=info ACCELERATE_USE_DEEPSPEED=false \
        accelerate launch \
        --num_processes "$NUM_PROCESSES" \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        src/open_r1/eval_gsm8k_rewards.py "${ARGS[@]}"
fi
