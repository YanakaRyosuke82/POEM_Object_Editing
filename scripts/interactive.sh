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


if [ "$1" = "DRAW" ]; then
    python src/main_draw.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen \
        --is_benchmark_dataset \
        --max_objects 5 \
        --draw
elif [ "$1" = "DRAW_SKIP_440" ]; then
    python src/main_draw_skip_500.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen \
        --is_benchmark_dataset \
        --max_objects 5 \
        --draw --skip_samples 440
elif [ "$1" = "DRAW_SKIP_840" ]; then
    python src/main_draw_skip_500.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen \
        --is_benchmark_dataset \
        --max_objects 5 \
        --draw --skip_samples 840

elif [ "$1" = "BENCHMARK_VLM_QWEN_MATH_DEEPSEEK" ]; then
    python src/main.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp1/output_grounding_dino_qwen_vlm_math_deepseek \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --vlm_model_name qwen_2_5_vl_7b \
        --math_llm_name deepseek_r1_distill_qwen_32B 

elif [ "$1" = "BENCHMARK_VLM_QWEN_MATH_DEEPSEEK" ]; then
    python src/main.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp1/output_grounding_dino_qwen_vlm_math_deepseek \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --vlm_model_name qwen_2_5_vl_7b \
        --math_llm_name deepseek_r1_distill_qwen_32B \
        --skip_samples 400

elif [ "$1" = "BENCHMARK_VLM_QWEN_MATH_QWEN" ]; then
    python src/main_continue_exp2_GD.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --vlm_model_name qwen_2_5_vl_7b \
        --math_llm_name qwen2_5_math_7b_instruct

elif [ "$1" = "BENCHMARK_VLM_INTERN_MATH_DEEPSEEK" ]; then
    python src/main_continue_exp3.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp3/output_vlm_intern_math_deepseek \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --mode self_correction \
        --vlm_model_name intern_vl_2_5_8B \
        --math_llm_name deepseek_r1_distill_qwen_32B

elif [ "$1" = "BENCHMARK_VLM_INTERN_MATH_QWEN" ]; then
    python src/main.py \
        --in_dir benchmark/input \
        --out_dir benchmark/exp4/output_vlm_intern_math_qwen \
        --is_benchmark_dataset \
        --max_objects 5 \
        --reasoning \
        --vlm_model_name intern_vl_2_5_8B \
        --math_llm_name qwen2_5_math_7b_instruct

elif [ "$1" = "BREAKING_POINT" ]; then
    python src/main.py \
        --in_dir exp_breaking_point_3/input \
        --out_dir exp_breaking_point_3/output \
        --max_objects 5 \
        --reasoning \
        --draw \
        --vlm_model_name qwen_2_5_vl_7b \
        --math_llm_name qwen2_5_math_7b_instruct
else
    echo "Invalid argument"
fi

