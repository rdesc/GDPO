#!/bin/bash

#SBATCH --partition=short-unkillable
#SBATCH -c 24                                                           
#SBATCH --mem=128G                                        
#SBATCH --time=3:00:00   
#SBATCH --requeue   
#SBATCH --gres=gpu:a100l:4           
#SBATCH --job-name=cgrpo_gsm8k_qwen2.5-1.5B
#SBATCH -o /home/mila/g/girgisro/git_repos/GDPO/slurm_logs/cgrpo_training/%x.out

module load anaconda/3
module load cudatoolkit/12.4.0

conda activate gdpo-trl

cd /home/mila/g/girgisro/git_repos/GDPO/trl-GDPO/open-r1/

HF_HUB_OFFLINE=1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/cgrpo_gsm8k/config.yaml \
    --vllm_mode colocate \
    --use_constraints True \
    --constraints_thresholds 0.96 0.97 0.95 \
    --update_constraints_every_k_policy_steps 1 \
    --max_length_threshold 600