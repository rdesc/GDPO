#!/bin/bash

#SBATCH --partition=short-unkillable
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:a100l:4
#SBATCH --job-name=ppo_gsm8k_qwen2.5-1.5B
#SBATCH -o /home/mila/g/girgisro/git_repos/GDPO/slurm_logs/ppo_training/%x.out

module load anaconda/3
module load cudatoolkit/12.4.0

conda activate gdpo-trl

cd /home/mila/g/girgisro/git_repos/GDPO/trl-GDPO/open-r1/

HF_HUB_OFFLINE=1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/ppo_gsm8k/config.yaml \
    --vllm_mode colocate \
    --use_ppo True \
    --gae_gamma 1.0 \
    --gae_lambda 0.95 \
    --value_model_lr 1e-4 \
    --value_loss_coef 0.5 \
    --value_warmup_steps 10
