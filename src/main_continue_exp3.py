import os
import logging
import argparse
from typing import Optional
import subprocess
import time

import cv2
import torch
import numpy as np
from tqdm import tqdm


# Custom utils imports
from utils_pose.models import Models
from utils_pose.open_cv_transformations import run_open_cv_transformations
from utils_pose.sam_refiner import run_sam_refine
from utils_pose.math_model import run_math_analysis
from utils_pose.vlm_image_parser import parse_image, save_results_image_parse
from utils_pose.sld_adapter import generate_sld_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_logging() -> logging.Logger:
    """Configure and return logger with colored output for model loading and standard output for other logs.

    Returns:
        logging.Logger: Configured logger instance for the main module
    """
    # Set up colored logger for model loading
    model_loader_logger = logging.getLogger("model_loader")
    model_handler = logging.StreamHandler()
    color_formatter = logging.Formatter("\033[36m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m")
    model_handler.setFormatter(color_formatter)
    model_loader_logger.addHandler(model_handler)
    model_loader_logger.setLevel(logging.INFO)

    # Set up colored logging for other modules
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    color_formatter = logging.Formatter("\033[36m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m")
    handler.setFormatter(color_formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def run_sld(
    json_path: str,
    input_path: str,
    output_dir: str,
    logger: logging.Logger,
    NORMAL_GPU: str,
    evaluation_folder_before: str,
    evaluation_folder_refined: str,
    save_file_name: str,
    mode: str,
) -> None:
    """
    Run the SLD (Structure-aware Latent Diffusion) pipeline
    """
    ext_script = "src/SLD/SLD_demo.py"
    config_ini = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/src/SLD/demo_config.ini"

    evaluation_before_path = os.path.abspath(evaluation_folder_before)
    evaluation_refined_path = os.path.abspath(evaluation_folder_refined)

    os.environ["CUDA_VISIBLE_DEVICES"] = NORMAL_GPU.replace("cuda:", "")
    subprocess.run(
        [
            "python",
            ext_script,
            "--json-file",
            json_path,
            "--input-dir",
            input_path,
            "--output-dir",
            output_dir,
            "--mode",
            mode,
            "--config",
            config_ini,
            "--evaluation-path-before",
            evaluation_before_path,
            "--evaluation-path-refined",
            evaluation_refined_path,
            "--save-file-name",
            save_file_name,
        ]
    )


def save_run_details(args, logger, output_path=None, timing_stats=None):
    """Save important details about the run to a text file"""
    if output_path is None:
        output_path = os.path.join(args.out_dir, "run_details.txt")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if timing_stats is None:
        # Initial configuration
        details = [
            "=== Run Details ===",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nConfiguration:",
            f"- Input directory: {args.in_dir}",
            f"- Output directory: {args.out_dir}",
            f"- Edit mode: {args.edit}",
            f"- Drawing enabled: {args.draw}",
            f"- Reasoning enabled: {args.reasoning}",
            f"- Max objects: {args.max_objects}",
            f"- Is benchmark dataset: {args.is_benchmark_dataset}",
            f"- Mode: {args.mode}",
            f"- VLM model: {args.vlm_model_name}",
            f"- Math LLM model: {args.math_llm_name}",
        ]

        try:
            with open(output_path, "w") as f:
                f.write("\n".join(details))
            logger.info(f"Initial run configuration saved to {output_path}")
        except FileNotFoundError as e:
            logger.error(f"Failed to save run details: {e}")

    else:
        # Append timing statistics
        details = [
            f"\n=== Performance Metrics ===",
            f"- Total samples processed: {timing_stats['sample_count']}",
            f"- Total runtime: {timing_stats['total_time']:.2f}s",
            f"- Average time per sample: {timing_stats['avg_total']:.2f}s",
        ]

        if args.reasoning:
            details.extend(
                [f"- Total reasoning time: {timing_stats['reasoning_time']:.2f}s", f"- Average reasoning time per sample: {timing_stats['avg_reasoning']:.2f}s"]
            )

        if args.draw:
            details.extend(
                [f"- Total drawing time: {timing_stats['drawing_time']:.2f}s", f"- Average drawing time per sample: {timing_stats['avg_drawing']:.2f}s"]
            )

        try:
            with open(output_path, "a") as f:
                f.write("\n".join(details))
            logger.info(f"Performance metrics appended to {output_path}")
        except FileNotFoundError as e:
            logger.error(f"Failed to append performance metrics: {e}")


def main():
    """Main entry point of the program"""
    logger = setup_logging()

    # Set up CUDA devices
    DEEP_SEEK_GPU = "cuda:1" if torch.cuda.is_available() else "cpu"
    NORMAL_GPU = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEEP_SEEK_GPU}")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process images with edit instructions")
    parser.add_argument("--in_dir", type=str, required=True, help="Input images directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--edit", type=str, choices=["resize", "grayscale", "none"], default="none")
    parser.add_argument("--draw", action="store_true", help="Enable drawing mode")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode")
    parser.add_argument("--max_objects", type=int, default=5, help="Maximum number of objects allowed to be in an image")
    parser.add_argument("--is_benchmark_dataset", action="store_true", help="Enable benchmark dataset mode")
    parser.add_argument("--mode", type=str, choices=["self_correction", "image_editing"], default="image_editing", help="Mode to run the pipeline in")
    parser.add_argument(
        "--vlm_model_name",
        type=str,
        choices=["qwen_2_5_vl_7b", "intern_vl_2_5_8B"],
        default="qwen_2_5_vl_7b",
        help="VLM model to use",
    )
    parser.add_argument(
        "--math_llm_name",
        type=str,
        choices=["deepseek_r1_distill_qwen_32B", "qwen2_5_math_7b_instruct"],
        default="deepseek_r1_distill_qwen_32B",
        help="Math LLM model to use",
    )
    args = parser.parse_args()

    # save config details to txt
    save_run_details(args=args, logger=logger)

    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    evaluation_folders = {
        "evaluation_1_after_vlm": os.path.join(args.out_dir, "evaluation_1_after_vlm"),
        "evaluation_2_after_sam": os.path.join(args.out_dir, "evaluation_2_after_sam"),
        "evaluation_3_after_llm_transformation": os.path.join(args.out_dir, "evaluation_3_after_llm_transformation"),
        "evaluation_4_after_sld": os.path.join(args.out_dir, "evaluation_4_after_sld"),
        "evaluation_5_after_sld_refine": os.path.join(args.out_dir, "evaluation_5_after_sld_refine"),
        "evaluation_6_after_llm_transformatio_oracle": os.path.join(args.out_dir, "evaluation_6_after_llm_transformation_oracle"),
    }
    for folder in evaluation_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Load models
    models = Models(device_reasoning=NORMAL_GPU, DEEP_SEEK_GPU=DEEP_SEEK_GPU)
    if args.vlm_model_name == "qwen_2_5_vl_7b":
        vlm_model, vlm_processor = models.get_qwen_2_5_vl_7b()
    elif args.vlm_model_name == "intern_vl_2_5_8B":
        vlm_model = models.get_intern_vl_2_5_8B()
        vlm_processor = None
    sam_model = models.get_sam()
    if args.math_llm_name == "deepseek_r1_distill_qwen_32B":
        math_model, math_tokenizer = models.get_deepseek_r1_distill_qwen_32B()
    elif args.math_llm_name == "qwen2_5_math_7b_instruct":
        math_model, math_tokenizer = models.get_qwen2_5_math_7b_instruct()

    # Initialize time tracking variables
    start_time = time.time()
    reasoning_time = 0
    drawing_time = 0
    sample_count = 0

    # count number of folder sin args.in_dir and print it
    num_in_folders = len(os.listdir(args.in_dir))
    logger.info(f"Number of folders in {args.in_dir}: {num_in_folders}")

    isoccured = False

    # Process each image in input directory
    total_samples = sum(1 for _, _, files in os.walk(args.in_dir) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")) and f != "input_mask.png")
    progress_bar = tqdm(total=total_samples, desc="Processing samples")

    for sample_idx, (subdir, _, files) in enumerate(os.walk(args.in_dir)):
        for filename in files:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            if filename == "input_mask.png":
                continue

            if args.is_benchmark_dataset:
                with open(os.path.join(subdir, "save_file_name.txt"), "r") as file:
                    save_file_name = file.read().strip()
                    if save_file_name == "2008_002379_0_transform_1_prompt_2":
                        isoccured = True

            else:
                save_file_name = os.path.basename(subdir).strip()
            sample_count += 1
            if not isoccured:
                continue

            # Set up paths
            input_path = os.path.join(subdir, filename)
            subfolder_name = os.path.basename(subdir)
            sample_dir = os.path.join(args.out_dir, subfolder_name)
            os.makedirs(sample_dir, exist_ok=True)

            analysis_file = os.path.join(sample_dir, "analysis.txt")
            analysis_enhanced_file = os.path.join(sample_dir, "analysis_enhanced.txt")
            transformation_matrix_file = os.path.join(sample_dir, "transformation_matrix.npy")
            json_path = os.path.join(sample_dir, "config_sld.json")
            edit_instruction_file = os.path.join(subdir, "edit_instruction.txt")

            with open(edit_instruction_file, "r") as file:
                USER_EDIT = file.read().strip()

            logger.info(f"Processing sample #{sample_idx}: {input_path}")
            ### REASONING ###
            reasoning_start = time.time()
            if args.reasoning:
                #  STEP 1 VLM ---  parsing
                logger.info(f"Step 1: VLM Parsing for sample {sample_idx}/{num_in_folders}")
                try:
                    results = parse_image(input_path, args.vlm_model_name, vlm_model, vlm_processor, NORMAL_GPU, USER_EDIT)
                    save_results_image_parse(sample_dir, results)
                except:
                    logger.error(f"No VLM parsing found for sample {sample_idx}")
                    continue
                VLM_BBOXES = results["objects"]
                if len(VLM_BBOXES) > args.max_objects:
                    logger.error(f"Too many objects detected for sample {sample_idx}/{num_in_folders}")
                    continue

                # Step 2: Refine detections with SAM
                logger.info(f"Step 2: SAM Refine Detections for sample {sample_idx}")
                try:
                    SAM_MASKS = run_sam_refine(file_analysis_path=analysis_file, img_path=input_path, sam_model=sam_model)
                except:
                    logger.error(f"No SAM masks found for sample {sample_idx}")
                    continue

                # Step 3:  LLM: Mathematical analysis
                logger.info(f"Step 3 - LLM Math Analysis for sample {sample_idx}/{num_in_folders}")
                try:
                    _, OBJECT_ID = run_math_analysis(
                        user_edit=USER_EDIT,
                        file_path=analysis_enhanced_file,
                        model_name=args.math_llm_name,
                        model=math_model,
                        tokenizer=math_tokenizer,
                        device=DEEP_SEEK_GPU,
                        logger=logger,
                    )
                except:
                    logger.error(f"No math analysis found for object {OBJECT_ID}")
                    continue

                # Step 4: Apply transformations
                logger.info(f"Step 4: OPEN-CV Transformations for sample {sample_idx}/{num_in_folders}")
                try:
                    TRANSFORMED_MASK, TRANSFORMED_ORACLE = run_open_cv_transformations(
                        matrix_transform_file=transformation_matrix_file,
                        output_dir=sample_dir,
                        oracle_mask_path=os.path.join(subdir, "input_mask.png"),
                        ENHANCED_FILE_DESCRIPTION=analysis_enhanced_file,
                    )
                except:
                    logger.error(f"No transformation matrix found for object {OBJECT_ID}")
                    # Create a black image with the same dimensions as the input image
                    TRANSFORMED_MASK = np.zeros((512, 512))
                    continue

                # Get masks and bounding boxes
                try:
                    VLM_BBOX = VLM_BBOXES[OBJECT_ID - 1]["bbox"]
                except:
                    logger.error(f"No bounding box found for object {OBJECT_ID}")
                    # Create a black mask with same dimensions as SAM_MASKS
                    VLM_BBOX = [0, 0, 1, 1]
                    continue

                try:
                    SAM_MASK = SAM_MASKS[str(OBJECT_ID)].astype(np.uint8) * 255
                except:
                    logger.error(f"No SAM mask found for object {OBJECT_ID}")
                    # Create a black image with the same dimensions as the input image
                    SAM_MASK = np.zeros((512, 512))
                    continue

                # Save evaluation results
                cv2.imwrite(os.path.join(evaluation_folders["evaluation_2_after_sam"], f"{save_file_name}.png"), SAM_MASK)
                cv2.imwrite(os.path.join(evaluation_folders["evaluation_3_after_llm_transformation"], f"{save_file_name}.png"), TRANSFORMED_MASK)
                cv2.imwrite(os.path.join(evaluation_folders["evaluation_6_after_llm_transformatio_oracle"], f"{save_file_name}.png"), TRANSFORMED_ORACLE)

                # Create and save VLM mask
                height, width = SAM_MASK.shape
                vlm_mask = np.zeros((height, width), dtype=np.uint8)
                xmin, ymin = int(VLM_BBOX[0] * width), int(VLM_BBOX[1] * height)
                xmax, ymax = int(VLM_BBOX[2] * width), int(VLM_BBOX[3] * height)
                vlm_mask[ymin:ymax, xmin:xmax] = 255
                cv2.imwrite(os.path.join(evaluation_folders["evaluation_1_after_vlm"], f"{save_file_name}.png"), vlm_mask)

                # Step 5: Generate config_sld.json for the SLD
                logger.info(f"Step 5: SLD Generation for sample {sample_idx}/{num_in_folders}")
                generate_sld_config(sample_dir, analysis_enhanced_file)

                reasoning_time += time.time() - reasoning_start

            ### IMAGE GENERATION ###
            drawing_start = time.time()
            if args.draw:

                # Step 6: Run SLD to generate edited image
                logger.info(f"Step 6: SLD Generation for sample {sample_idx}/{num_in_folders}")
                run_sld(
                    json_path=os.path.abspath(json_path),
                    input_path=os.path.abspath(input_path),
                    output_dir=os.path.abspath(sample_dir),
                    logger=logger,
                    NORMAL_GPU=NORMAL_GPU,
                    evaluation_folder_before=evaluation_folders["evaluation_4_after_sld"],
                    evaluation_folder_refined=evaluation_folders["evaluation_5_after_sld_refine"],
                    save_file_name=save_file_name,
                    mode=args.mode,
                )
            drawing_time += time.time() - drawing_start

            progress_bar.update(1)

    progress_bar.close()

    # After the main processing loop, before the timing statistics logging:
    total_time = time.time() - start_time
    timing_stats = {
        "sample_count": sample_count,
        "total_time": total_time,
        "reasoning_time": reasoning_time,
        "drawing_time": drawing_time,
        "avg_total": total_time / sample_count if sample_count > 0 else 0,
        "avg_reasoning": reasoning_time / sample_count if sample_count > 0 and args.reasoning else 0,
        "avg_drawing": drawing_time / sample_count if sample_count > 0 and args.draw else 0,
    }
    save_run_details(args=args, logger=logger, timing_stats=timing_stats)

    # Log timing statistics
    if sample_count > 0:
        logger.info(f"Average processing time per sample: {timing_stats['avg_total']:.2f}s")
        if args.reasoning:
            logger.info(f"Average reasoning time per sample: {timing_stats['avg_reasoning']:.2f}s")
        if args.draw:
            logger.info(f"Average drawing time per sample: {timing_stats['avg_drawing']:.2f}s")


if __name__ == "__main__":
    main()
