import torch
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
import transformers
print(f"transformers {transformers.__version__}")
import trl
print(f"trl {trl.__version__}")
import vllm
print(f"vllm {vllm.__version__}")
import flash_attn
print(f"flash-attn {flash_attn.__version__}")
import deepspeed
print(f"deepspeed {deepspeed.__version__}")
print("All imports OK!")
