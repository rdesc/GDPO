#!/bin/bash
#SBATCH --account=aip-pal 
#SBATCH --job-name=cgrpo-train
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/r/rogerg/GDPO/slurm_logs/%x.out

# ===== Load environment =====
module load StdEnv/2023
module load python/3.10
module load cuda/12.6   

source /home/r/rogerg/envs/gdpo-trl/bin/activate

wandb offline

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3_8gpu.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/cgrpo_gsm8k/config_8gpu.yaml \
    --vllm_mode colocate \
    --use_constraints True \
    --constraints_thresholds 0.96 0.97 0.95 \
    --update_constraints_every_k_policy_steps 1 \
    --max_length_threshold 600