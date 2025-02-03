source .venv2/bin/activate

module load cuda

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

python src/main_benchmark.py \
    --in_dir benchmark_results_500/input_benchmark \
    --out_dir benchmark_results_500/output_benchmark \
    --edit "grayscale" \
    --max_objects 5 \
    --dataset_size_samples 500 \
    --reasoning \
    --draw