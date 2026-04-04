#!/bin/bash

#SBATCH --partition=short-unkillable
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:a100l:4
#SBATCH --job-name=ppo_gsm8k_qwen2.5-1.5B
#SBATCH -o /home/mila/d/deschaer/GDPO/slurm_logs/ppo_training/%x_%j.out

module load anaconda/3
module load cudatoolkit/12.4.0

conda activate gdpo-trl

cd /home/mila/d/deschaer/GDPO/trl-GDPO/open-r1/

HF_HUB_OFFLINE=1 ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/gsm8k.py \
  --config recipes/Qwen2.5-1.5B-Instruct/ppo_gsm8k/config.yaml \
  --use_ppo True \
  --num_generations 8 \
  --vllm_mode colocate \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --beta 0.04 \
  --advantage_type reinforce \
  --value_warmup_steps 200 \
  --value_model_type separate \
  --value_model_use_lora True \
  --value_model_lora_rank 64 \
  --value_model_lora_alpha 128 \
  --value_model_lr 1e-5 \
  --run_name "PPO-reinforce-Vs0-beta0.04" \
  --output_dir /home/mila/d/deschaer/scratch/saves/rl/
