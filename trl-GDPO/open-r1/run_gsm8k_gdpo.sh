

WANDB_DIR=/home/mila/d/deschaer/scratch/gdpo-output HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3_1gpu.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/gdpo_gsm8k/config.yaml \
    --vllm_mode colocate

