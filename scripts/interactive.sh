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

if [ "$1" = "MARCO_BENCHMARK" ]; then
    python src/main.py \
        --in_dir benchmark_results_FULL/input_benchmark \
        --out_dir benchmark_results_FULL/output_benchmark \
        --edit "grayscale" \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --draw


elif [ "$1" = "breaking_point" ]; then
    python src/main.py \
        --in_dir exp_breaking_point/input \
        --out_dir exp_breaking_point/output \
        --edit "grayscale" \
        --reasoning \
        --draw

elif [ "$1" = "exp_breaking_point_easy_user_edit" ]; then
    python src/main.py \
        --in_dir exp_breaking_point_easy_user_edit/input \
        --out_dir exp_breaking_point_easy_user_edit/output \
        --edit "grayscale" \
        --reasoning \
        --draw

elif [ "$1" = "debug" ]; then
    python src/main.py \
        --in_dir input_debug \
        --out_dir output_debug \
        --edit "grayscale" \
        --reasoning \
        --draw \
        --mode self_correction

elif [ "$1" = "exp_breaking_point_3" ]; then
    python src/main.py \
        --in_dir exp_breaking_point_3/input \
        --out_dir exp_breaking_point_3/output \
        --edit "grayscale" \
        --reasoning \
        --draw \
        --mode self_correction

else
    echo "Invalid argument"
fi


