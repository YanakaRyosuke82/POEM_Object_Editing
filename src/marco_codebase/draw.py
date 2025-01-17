import argparse
from PIL import Image
import torch
import os
import numpy as np
import diffusers
import logging 

from ddim_inversion import ddim_inversion
from operations import add_operation, remove_operation, modify_operation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(image_path: str, mask_path: str, save_path: str, operation: str, operation_params: dict) -> None:
    """Process an image using diffusion models with a mask."""
    logging.info(f"Processing image: {image_path} with mask: {mask_path} and operation: {operation}")
    
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert('L')

    tmp_dir = os.path.join(os.path.dirname(image_path), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_image_path = os.path.join(tmp_dir, os.path.basename(image_path))
    tmp_mask_path = os.path.join(tmp_dir, os.path.basename(mask_path))
    image.save(tmp_image_path)
    mask.save(tmp_mask_path)

    # Run diffusion pipeline
    image_latents = get_inverted_latents(image_path)
    logging.info(f"Shape of inverted latents: {image_latents.shape}")

    modified_latents = run_operation(image_latents, operation, operation_params)
    output = run_forward_diffusion(modified_latents, mask)
    
    output.save(save_path)
    logging.info(f"Output saved to: {save_path}")


# step 1
def get_inverted_latents(image_path: str) -> torch.Tensor:
    """Convert image to latent representation via backward diffusion"""
    inverted_latents = ddim_inversion(image_path, num_steps=50, verify=True)
    return inverted_latents

# step 2
def run_operation(latents: torch.Tensor,  operation: str, operation_params: dict) -> torch.Tensor:
    """Apply operation to latent representation."""
    if operation == "addition":
        modified_latents = add_operation(latents, operation_params)
    elif operation == "remove":
        modified_latents = remove_operation(latents, operation_params)
    elif operation == "modify":
        modified_latents = modify_operation(latents, operation_params)
    else:
        raise ValueError(f"Invalid operation: {operation}")

    return modified_latents  

# step 3
def run_forward_diffusion(latents: torch.Tensor, mask: Image) -> Image:
    """Convert latents back to image."""
    return Image.new('RGB', (256, 256))  # Placeholder

def main():
    parser = argparse.ArgumentParser(description='Process images using diffusion models with masks')
    parser.add_argument('--source_image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--source_mask_path', type=str, required=True, help='Path to mask image')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output image')
    parser.add_argument('--operation', type=str, required=True, help='Operation to perform (addition, remove, modify)')
    parser.add_argument('--operation_params', type=str, required=True, help='Operation parameters as JSON string')

    args = parser.parse_args()
    
    # Parse operation params from JSON string
    import json
    try:
        operation_params = json.loads(args.operation_params)
    except json.JSONDecodeError:
        parser.error("operation_params must be a valid JSON string")

    logging.info(f"Image Path: {args.source_image_path}")
    logging.info(f"Mask Path: {args.source_mask_path}")
    logging.info(f"Save Path: {args.save_path}")
    logging.info(f"Operation: {args.operation}")
    logging.info(f"Operation Params: {operation_params}")

    process_image(args.source_image_path, args.source_mask_path, args.save_path, args.operation, operation_params)

if __name__ == "__main__":
    main()
