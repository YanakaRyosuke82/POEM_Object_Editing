#!/bin/sh
#BSUB -q gpuh100
#BSUB -J "ip2p"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o error/%J.out
#BSUB -e error/%J.err


module load python3/3.10.12
module load cuda

source env/bin/activate



# Define the cache path
CACHE_PATH="/dtu/blackhole/14/189044/marscho/cache/"

# Ensure the directory exists
mkdir -p "$CACHE_PATH/tmp"

# Export environment variables
export HF_HOME="$CACHE_PATH"
export TRANSFORMERS_CACHE="$CACHE_PATH"
export DATASETS_CACHE="$CACHE_PATH"
export TORCH_HOME="$CACHE_PATH"
export TORCH_CACHEDIR="$CACHE_PATH"
export CUDA_CACHE_PATH="$CACHE_PATH"
export WANDB_CACHE_DIR="$CACHE_PATH"
export WANDB_DATA_DIR="$CACHE_PATH"
export PYTHONPYCACHEPREFIX="$CACHE_PATH"
export TMPDIR="$CACHE_PATH/tmp"

# for easier to read terminal errors
export BETTER_EXCEPTIONS=1



python3 edit_app.py
