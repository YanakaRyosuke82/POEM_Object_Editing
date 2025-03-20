import torch
import sys

def check_torch_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")
        
        # Test CUDA with a simple operation
        x = torch.rand(5, 3)
        print("\nTesting CUDA with a simple tensor operation:")
        print(f"Tensor on CPU: {x}")
        x = x.cuda()
        print(f"Tensor on GPU: {x}")
        print("CUDA test successful!")
    else:
        print("\nWARNING: CUDA is not available. PyTorch will only use CPU.")

if __name__ == "__main__":
    check_torch_cuda() 