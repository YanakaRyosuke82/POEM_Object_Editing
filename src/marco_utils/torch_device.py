import torch

def get_device():
    """Get the appropriate torch device"""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"

device = get_device() 