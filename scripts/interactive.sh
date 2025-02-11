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

if [ "$1" = "BENCHMARK_VLM_QWEN_MATH_DEEPSEEK" ]; then
    python src/main.py \
        --in_dir benchmark/input \
        --out_dir benchmark/output_math_deepseek \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --draw \
        --vlm_model_name qwen_2_5_vl_7b \
        --math_llm_name deepseek_r1_distill_qwen_32B

elif [ "$1" = "BENCHMARK_VLM_QWEN_MATH_QWEN" ]; then
    python src/main.py \
        --in_dir benchmark/input \
        --out_dir benchmark/output_math_qwen \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --draw \
        --vlm_model_name qwen_2_5_vl_7b \
        --math_llm_name qwen2_5_math_7b_instruct

elif [ "$1" = "BENCHMARK_VLM_OVIS_GEMMA2_MATH_DEEPSEEK" ]; then
    python src/main.py \
        --in_dir benchmark/input \
        --out_dir benchmark/output_vlm_ovis_gemma2 \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --draw \
        --vlm_model_name ovis1_6_gemma2_27B \
        --math_llm_name deepseek_r1_distill_qwen_32B
else
    echo "Invalid argument"
fi

