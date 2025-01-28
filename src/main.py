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
from marco_utils.qwen_math import run_math_analysis
from marco_utils.vlm_image_parser import parse_image, save_results_image_parse
from marco_utils.sld_adapter import generate_sld_config
from marco_utils.torch_device import device  # Import the device


# # SLD imports
# from SLDclean.SLD_demo import main as sld_main

def setup_logging() -> logging.Logger:
    """Configure and return logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
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

def run_sld(json_path: str, input_path: str, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the SLD (Structure-aware Latent Diffusion) pipeline
    """
    # Path to the external script
    ext_script = "src/SLD/SLD_demo.py"
    config_ini = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/src/SLD/demo_config.ini"
        # Run the script
    subprocess.run([
        "python", ext_script,
        '--json-file', json_path,
        '--input-dir', input_path,
        '--output-dir', output_dir,
        '--mode', 'image_editing',
        '--config', config_ini
    ])


def main():
    """Main entry point of the program"""
    logger = setup_logging()

    # Check for CUDA availability and select device
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Initialize models
    models = Models(device)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process images with edit instructions')
    parser.add_argument('--in_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--out_dir', type=str, required=True, 
                      help='Directory for output files')
    parser.add_argument('--edit', type=str, choices=['resize', 'grayscale', 'none'],
                      default='none', help='Edit instruction to apply to images')
    parser.add_argument('--draw', action='store_true', help='Enable drawing mode')
    parser.add_argument('--reasoning', action='store_true', help='Enable reasoning mode')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load models
    vlm_model, vlm_processor = models.get_qwen_vlm()
    sam_model = models.get_sam()
    math_model, math_tokenizer = models.get_qwen_math()
    # math_model, math_tokenizer = models.get_deepseek_r1_text()

    start_time = time.time()
    reasoning_time = 0
    drawing_time = 0
    sample_count = 0

    # Process each image in input directory
    for sample_idx, (subdir, _, files) in enumerate(os.walk(args.in_dir)):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            sample_count += 1

            ## set paths    
            input_path = os.path.join(subdir, filename)
            sample_dir = os.path.join(args.out_dir, f"sample_{sample_idx:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            analysis_file = os.path.join(sample_dir, "analysis.txt")
            analysis_enhanced_file = os.path.join(sample_dir, "analysis_enhanced.txt")
            transformation_matrix_file = os.path.join(sample_dir, "transformation_matrix.npy")
            json_path = os.path.join(sample_dir, "config_sld.json")
            edit_instruction_file = os.path.join(subdir, "edit_instruction.txt")
            with open(edit_instruction_file, 'r') as file:
                USER_EDIT = file.read().strip()
            
            logger.info(f"Processing sample {sample_idx}: {input_path}")
            try:
                # Step 1: Process image
                processed_image = process_image(input_path, args.edit)
                if processed_image is None:
                    continue
                ### REASONING ###
                reasoning_start = time.time()
                if args.reasoning:
                    # Step 2: Parse image for analysis
                    results = parse_image(input_path, vlm_model, vlm_processor, device, USER_EDIT)
                    save_results_image_parse(sample_dir, processed_image, input_path, results)
                    
                    # Step 3: Refine detections with SAM
                    logger.info(f"Refining detections for sample {sample_idx}")
                    run_sam_refine(
                        file_analysis_path=analysis_file,
                        img_path=input_path,
                        sam_model=sam_model
                    )
                    # Step 4: Mathematical analysis
                    logger.info(f"Performing mathematical analysis for sample {sample_idx}")
                    run_math_analysis(
                        user_edit=USER_EDIT,  # Now using the read edit instruction specific to this sample
                        file_path=analysis_enhanced_file,
                        img_path=input_path,
                        model=math_model,
                        tokenizer=math_tokenizer,
                        device=device
                    )

                    # Step 5: Apply transformations
                    run_open_cv_transformations(
                        matrix_transform_file=transformation_matrix_file,
                        output_dir=sample_dir,
                        MASK_FILE_NAME="mask_0.png",
                        ENHANCED_FILE_DESCRIPTION=analysis_enhanced_file
                    )
                reasoning_time += time.time() - reasoning_start
                   
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
                        logger=logger
                    )
                drawing_time += time.time() - drawing_start

            except Exception as e:
                logger.error(f"Error processing {input_path}: {str(e)}")
                continue



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