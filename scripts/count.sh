for path in \
    "benchmark/exp1/output_grounding_dino_qwen_vlm_math_deepseek/evaluation_3_after_llm_transformation" \
    "benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen/evaluation_3_after_llm_transformation" \
    "benchmark/exp5/output_grounding_dino_vlm_only/evaluation_3_after_llm_transformation" \
    "benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen/evaluation_4_after_sld"
do
    echo -n "Experiment: "
    echo "$path" 
    echo -n "Count: "
    ls "$path" | wc -l
done
