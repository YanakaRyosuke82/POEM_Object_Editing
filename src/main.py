import os
import sys
import logging
import argparse
import shutil
from typing import Optional
import subprocess
import time


import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import configparser

# Custom utils imports
from marco_utils.models import Models
from marco_utils.open_cv_transformations import run_open_cv_transformations
from marco_utils.sam_refiner import run_sam_refine
from marco_utils.math_model import run_math_analysis
from marco_utils.vlm_image_parser import parse_image, save_results_image_parse
from marco_utils.sld_adapter import generate_sld_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# # SLD imports
# from SLDclean.SLD_demo import main as sld_main


def setup_logging() -> logging.Logger:
    """Configure and return logger"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def process_image(image_path: str, edit_type: str) -> Optional[cv2.Mat]:
    """
    Process image according to edit type

    Args:
        image_path: Path to input image
        edit_type: Type of edit to apply ('resize' or 'grayscale')

    Returns:
        Processed image or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if edit_type == "resize":
            return cv2.resize(image, (512, 512))
        elif edit_type == "grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None


def run_sld(
    json_path: str, input_path: str, output_dir: str, logger: logging.Logger, NORMAL_GPU: str, evaluation_folder_before: str, evaluation_folder_refined: str
) -> None:
    """
    Run the SLD (Structure-aware Latent Diffusion) pipeline
    """
    # Path to the external script
    ext_script = "src/SLD/SLD_demo.py"
    config_ini = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/src/SLD/demo_config.ini"
    # Run the script

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
            "image_editing",
            "--config",
            config_ini,
            "--evaluation-folder-before",
            evaluation_folder_before,
            "--evaluation-folder-refined",
            evaluation_folder_refined,
        ]
    )


def main():
    """Main entry point of the program"""
    logger = setup_logging()

    # Check for CUDA availability and select device
    if torch.cuda.is_available():
        DEEP_SEEK_GPU = "cuda:1"  # Select CUDA device 1
        NORMAL_GPU = "cuda:0"  # Select CUDA device 2

    else:
        DEEP_SEEK_GPU = "cpu"
        NORMAL_GPU = "cpu"
    logger.info(f"Using device: {DEEP_SEEK_GPU}")

    # Initialize models
    models = Models(device_reasoning=NORMAL_GPU, DEEP_SEEK_GPU=DEEP_SEEK_GPU)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process images with edit instructions")
    parser.add_argument("--in_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--edit", type=str, choices=["resize", "grayscale", "none"], default="none", help="Edit instruction to apply to images")
    parser.add_argument("--draw", action="store_true", help="Enable drawing mode")
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    evaluation_1_folder = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/evaluation_1_after_vlm"
    evaluation_2_folder = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/evaluation_2_after_sam"
    evaluation_3_folder = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/evaluation_3_after_llm_transformation"
    evaluation_4_folder = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/evaluation_4_after_sld"
    evaluation_5_folder = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/evaluation_5_after_sld_refine"
    os.makedirs(evaluation_1_folder, exist_ok=True)
    os.makedirs(evaluation_2_folder, exist_ok=True)
    os.makedirs(evaluation_3_folder, exist_ok=True)
    os.makedirs(evaluation_4_folder, exist_ok=True)
    os.makedirs(evaluation_5_folder, exist_ok=True)
    # Load models
    vlm_model, vlm_processor = models.get_qwen_vlm()
    sam_model = models.get_sam()
    # math_model, math_tokenizer = models.get_qwen_math()
    math_model, math_tokenizer = models.get_deepseek_r1_text()

    # time computations and logging
    start_time = time.time()
    reasoning_time = 0
    drawing_time = 0
    sample_count = 0

    # Process each image in input directory
    for sample_idx, (subdir, _, files) in enumerate(os.walk(args.in_dir)):
        for filename in files:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            sample_count += 1

            ## set paths
            input_path = os.path.join(subdir, filename)
            # Use the subfolder name from input dir as the output sample dir name
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

            processed_image = process_image(input_path, args.edit)
            if processed_image is None:
                continue
            ### REASONING ###
            reasoning_start = time.time()
            if args.reasoning:
                #  STEP 1 VLM ---  parsing
                try:
                    results = parse_image(input_path, vlm_model, vlm_processor, NORMAL_GPU, USER_EDIT)
                    save_results_image_parse(sample_dir, processed_image, input_path, results)
                except:
                    print(f"No VLM parsing found for sample {sample_idx}")
                    continue
                VLM_BBOXES = results["objects"]

                # Step 2: Refine detections with SAM
                logger.info(f"Refining detections for sample {sample_idx}")
                try:
                    SAM_MASKS = run_sam_refine(file_analysis_path=analysis_file, img_path=input_path, sam_model=sam_model)
                except:
                    print(f"No SAM masks found for sample {sample_idx}")
                    continue

                # Step 3:  LLM: Mathematical analysis
                logger.info(f"Performing mathematical analysis for sample {sample_idx}")
                try:
                    _, OBJECT_ID = run_math_analysis(
                        user_edit=USER_EDIT,  # Now using the read edit instruction specific to this sample
                        file_path=analysis_enhanced_file,
                        img_path=input_path,
                        model=math_model,
                        tokenizer=math_tokenizer,
                        device=DEEP_SEEK_GPU,
                    )
                except:
                    print(f"No math analysis found for object {OBJECT_ID}")
                    continue

                # Step 4: Apply transformations
                try:
                    TRANSFORMED_MASK = run_open_cv_transformations(
                        matrix_transform_file=transformation_matrix_file, output_dir=sample_dir, ENHANCED_FILE_DESCRIPTION=analysis_enhanced_file
                    )
                except:
                    print(f"No transformation matrix found for object {OBJECT_ID}")
                    # Create a black image with the same dimensions as the input image
                    TRANSFORMED_MASK = np.zeros_like(processed_image)
                    continue

                ### PREPARE FOR EVALUATION
                # SELECT THE CORRECT BBOX AND MASK
                try:
                    VLM_BBOX = VLM_BBOXES[OBJECT_ID - 1]["bbox"]
                except:
                    print(f"No bounding box found for object {OBJECT_ID}")
                    # Create a black mask with same dimensions as SAM_MASKS
                    VLM_BBOX = [0, 0, 1, 1]
                    continue

                try:
                    SAM_MASK = SAM_MASKS[str(OBJECT_ID)].astype(np.uint8) * 255
                except:
                    print(f"No SAM mask found for object {OBJECT_ID}")
                    # Create a black image with the same dimensions as the input image
                    SAM_MASK = np.zeros_like(processed_image)
                    continue

                # save the masks
                cv2.imwrite(os.path.join(evaluation_2_folder, f"mask_{sample_idx}.png"), SAM_MASK)
                cv2.imwrite(os.path.join(evaluation_3_folder, f"mask_{sample_idx}_transformed.png"), TRANSFORMED_MASK)
                # save the bboxes
                # Save VLM bbox as binary mask
                height, width = SAM_MASK.shape
                vlm_mask = np.zeros((height, width), dtype=np.uint8)
                xmin = int(VLM_BBOX[0] * width)
                ymin = int(VLM_BBOX[1] * height)
                xmax = int(VLM_BBOX[2] * width)
                ymax = int(VLM_BBOX[3] * height)
                vlm_mask[ymin:ymax, xmin:xmax] = 255
                cv2.imwrite(os.path.join(evaluation_1_folder, f"mask_{sample_idx}_vlm.png"), vlm_mask)

            ### IMAGE GENERATION ###
            drawing_start = time.time()
            if args.draw:
                # Step 6: Generate config_sld.json for the SLD
                generate_sld_config(sample_dir, analysis_enhanced_file)
                # Step 7: Run SLD to generate edited image
                run_sld(
                    json_path=os.path.abspath(json_path),
                    input_path=os.path.abspath(input_path),
                    output_dir=os.path.abspath(sample_dir),
                    logger=logger,
                    NORMAL_GPU=NORMAL_GPU,
                    evaluation_folder_before=evaluation_4_folder,
                    evaluation_folder_refined=evaluation_5_folder,
                )
            drawing_time += time.time() - drawing_start

    # time computations and logging
    end_time = time.time()
    total_time = end_time - start_time

    if sample_count > 0:
        avg_total_time = total_time / sample_count
        avg_reasoning_time = reasoning_time / sample_count if args.reasoning else 0
        avg_drawing_time = drawing_time / sample_count if args.draw else 0

        logger.info(f"Average processing time per sample: {avg_total_time:.2f} seconds")
        if args.reasoning:
            logger.info(f"Average reasoning time per sample: {avg_reasoning_time:.2f} seconds")
        if args.draw:
            logger.info(f"Average drawing time per sample: {avg_drawing_time:.2f} seconds")


if __name__ == "__main__":
    main()
