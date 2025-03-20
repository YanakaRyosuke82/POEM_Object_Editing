import torch
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", torch_dtype=torch.bfloat16)
pipe(messages)
breakpoint()
