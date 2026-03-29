
HF_HUB_OFFLINE=1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3_1gpu.yaml \
    src/open_r1/gsm8k.py --config recipes/Qwen2.5-1.5B-Instruct/cgrpo_gsm8k/config.yaml \
    --vllm_mode colocate \
    --use_constraints True \
    --constraints_thresholds 0.96 0.97 0.95 \
    --update_constraints_every_k_policy_steps 1 \
    --max_length_threshold 600 \
    --run_name cgrpo_gsm8k_qwen2.5-1.5B_1gpu_debug



