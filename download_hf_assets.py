"""Download HF model and dataset for offline use on compute nodes.
Run this on a login node (has internet access):
    module load python/3.10 cuda/12.6 cudnn
    source ~/envs/gdpo-trl/bin/activate
    python download_hf_assets.py
"""
from huggingface_hub import snapshot_download
from datasets import load_dataset

print("Downloading Qwen/Qwen2.5-1.5B-Instruct...")
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
print("Done.")

print("Downloading openai/gsm8k...")
load_dataset("openai/gsm8k", "main")
print("Done.")

print("All assets cached! You can now run with HF_HUB_OFFLINE=1.")
