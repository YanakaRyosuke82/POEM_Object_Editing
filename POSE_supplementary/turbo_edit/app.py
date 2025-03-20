from __future__ import annotations

import os
import argparse
import logging
from PIL import Image
import torch

from my_run import run as run_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_image(input_image_path, src_prompt, tgt_prompt, output_path, seed=7865, w1=1.5, w2=1.0):
    """Process a single image using the model"""
    try:
        logger.info(f"Processing: {input_image_path}")
        logger.info(f"Source prompt: {src_prompt}")
        logger.info(f"Target prompt: {tgt_prompt}")
        logger.info(f"Using seed: {seed}, w1: {w1}, w2: {w2}")

        # Run the model
        result_image = run_model(
            input_image_path,
            src_prompt,
            tgt_prompt,
            seed,
            w1,
            w2
        )

        # Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_image.save(output_path)
        logger.info(f"Result saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {input_image_path}: {e}")
        return False

def process_folder(input_folder, output_dir="output", seed=7865, w1=1.5, w2=1.0):
    """Process all sample folders with their respective instructions"""
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    # Collect all valid sample folders
    for sample_folder in os.scandir(input_folder):
        if not sample_folder.is_dir():
            continue

        sample_path = sample_folder.path
        input_image_path = os.path.join(sample_path, "input_image.png")
        instruction_file = os.path.join(sample_path, "edit_instruction.txt")
        output_sample_dir = os.path.join(output_dir, sample_folder.name)
        output_path = os.path.join(output_sample_dir, "output_image.png")

        # Skip if files don't exist or output already exists
        if not os.path.exists(input_image_path) or not os.path.exists(instruction_file):
            logger.warning(f"Missing files in {sample_path}, skipping")
            continue

        if os.path.exists(output_path):
            logger.info(f"Output already exists for {sample_path}, skipping")
            continue

        # Read instruction
        try:
            with open(instruction_file, 'r') as f:
                instruction = f.read().strip()

            if not instruction:
                logger.warning(f"Empty instruction in {instruction_file}, skipping")
                continue

            # Use the provided instruction for the image
            src_prompt = "a photo"  # Default source prompt
            tgt_prompt = instruction

            tasks.append((input_image_path, src_prompt, tgt_prompt, output_path, seed, w1, w2))
        except Exception as e:
            logger.error(f"Error reading instruction from {instruction_file}: {e}")

    # Process tasks
    if not tasks:
        logger.info("No tasks to process")
        return

    # Process images sequentially
    logger.info(f"Processing {len(tasks)} images sequentially")

    successful_count = 0
    for task in tasks:
        if process_image(*task):
            successful_count += 1

    logger.info(f"Completed processing {successful_count} of {len(tasks)} images successfully")

def main():
    parser = argparse.ArgumentParser(description="Turbo Edit")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing sample folders with images and instructions")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save edited images")
    parser.add_argument("--seed", type=int, default=7865, help="Random seed")
    parser.add_argument("--w1", type=float, default=1.5, help="Weight parameter")

    args = parser.parse_args()

    # Fixed w2 value as in the original code
    w2 = 1.0

    # Process the folder with instructions from each sample folder
    process_folder(
        input_folder=args.input_folder,
        output_dir=args.output_folder,
        seed=args.seed,
        w1=args.w1,
        w2=w2
    )

if __name__ == "__main__":
    main()
