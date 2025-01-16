import cv2
import numpy as np
from ddim_inversion import ddim_inversion
import torch
import os
from torchvision import transforms as tvt
import logging
import pdb
from typing import Union, Tuple, Optional
import logging

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDIMInverseScheduler,
    AutoencoderKL,
    DDIMScheduler
)
# pdb.set_trace()


def test_modify_object(image_path: str, transform_matrix: np.ndarray, mask_path: str) -> np.ndarray:
    """Apply transformation to object within mask region.
    
    Args:
        image_path: Path to input image
        transform_matrix: 3x3 transformation matrix
        mask_path: Path to binary mask image
        
    Returns:
        Modified image array at 64x64 scale
    """

    latents_history = ddim_inversion(image_path, num_steps=50, verify=True) # 50,1,4,64,64

    # Load image and mask
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    # Scale down both image and mask to 64x64
    SIZE_IMG = 64
    LATENT_SIZE = 64
    scaled_image = cv2.resize(image, (SIZE_IMG, SIZE_IMG))
    scaled_mask = cv2.resize(mask, (SIZE_IMG, SIZE_IMG), interpolation=cv2.INTER_NEAREST)
    latent_mask = cv2.resize(mask, (LATENT_SIZE, LATENT_SIZE), interpolation=cv2.INTER_NEAREST)

    # Create result image with gaussian noise background
    result = np.random.normal(128, 30, (SIZE_IMG, SIZE_IMG, 3)).astype(np.uint8)
    
    # Copy non-masked region from original image
    result = np.where(scaled_mask[:, :, np.newaxis] == 0, scaled_image, result)

    # Calculate object center for transformation using mask centroid
    moments = cv2.moments(scaled_mask)
    if moments["m00"] != 0:
        x_center = int(moments["m10"] / moments["m00"])
        y_center = int(moments["m01"] / moments["m00"])
    else:
        x_center = SIZE_IMG // 2
        y_center = SIZE_IMG // 2

    # Calculate latent space center
    latent_x_center = int(x_center * LATENT_SIZE / SIZE_IMG)
    latent_y_center = int(y_center * LATENT_SIZE / SIZE_IMG)

    # Create translation matrices for centering
    T_to_origin = np.array([
        [1, 0, -x_center],
        [0, 1, -y_center],
        [0, 0, 1]
    ], dtype=np.float32)

    T_from_origin = np.array([
        [1, 0, x_center],
        [0, 1, y_center],
        [0, 0, 1]
    ], dtype=np.float32)

    # Create translation matrices for latent space
    latent_T_to_origin = np.array([
        [1, 0, -latent_x_center],
        [0, 1, -latent_y_center],
        [0, 0, 1]
    ], dtype=np.float32)

    latent_T_from_origin = np.array([
        [1, 0, latent_x_center],
        [0, 1, latent_y_center],
        [0, 0, 1]
    ], dtype=np.float32)

    # Compose final transformations
    transform_matrix = transform_matrix.astype(np.float32)
    final_transform = T_from_origin @ transform_matrix @ T_to_origin
    final_latent_transform = latent_T_from_origin @ transform_matrix @ latent_T_to_origin

    # Apply transformation to full image
    transformed_image = cv2.warpPerspective(
        scaled_image,
        final_transform,
        (SIZE_IMG, SIZE_IMG),
        borderValue=(
            int(np.random.normal(128, 30)),
            int(np.random.normal(128, 30)),
            int(np.random.normal(128, 30))
        )
    )
    
    # Transform mask to identify valid transformed pixels
    transformed_mask = cv2.warpPerspective(
        scaled_mask,
        final_transform,
        (SIZE_IMG, SIZE_IMG),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )


    # Combine transformed region with result using masks
    result = np.where(transformed_mask[:, :, np.newaxis] > 0, transformed_image, result)



    # UPDATES LATENTS_HISTORY 
    # Transform latent mask
    transformed_latent_mask = cv2.warpPerspective(
        latent_mask,
        final_latent_transform,
        (LATENT_SIZE, LATENT_SIZE),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    # Process each timestep and channel in latents_history
    for t in range(latents_history.shape[0]):
        for c in range(latents_history.shape[2]):
            # Get current latent channel and normalize to 0-255 range
            latent_channel = latents_history[t, 0, c].detach().cpu().numpy()
            min_val = latent_channel.min()
            max_val = latent_channel.max()
            normalized_channel = ((latent_channel - min_val) / (max_val - min_val) * 255).round()
            normalized_channel = normalized_channel.astype(np.uint8)

            # Transform the masked region to new location
            transformed_region = cv2.warpPerspective(
                normalized_channel * (latent_mask > 0),  # Apply mask to select pixels to transform
                final_latent_transform,
                (LATENT_SIZE, LATENT_SIZE),
                borderValue=0
            )

            # Create mask of transformed pixels
            transformed_pixels = transformed_region > 0

            # Generate Gaussian noise for empty regions
            noise = np.random.normal(128, 30, normalized_channel.shape).astype(np.uint8)
            
            # First set original masked region to Gaussian noise instead of black
            result_channel = normalized_channel.copy()
            result_channel = np.where(latent_mask > 0, noise, result_channel)
            
            # Then overlay transformed values on top
            result_channel = np.where(transformed_pixels, transformed_region, result_channel)
            
            # Fill empty transformed areas with noise
            empty_mask = (transformed_latent_mask > 0) & (transformed_region == 0)
            result_channel = np.where(empty_mask, noise, result_channel)

            # Denormalize back to original range
            denormalized_channel = (result_channel / 255.0) * (max_val - min_val) + min_val
            
            # Update latents_history
            latents_history[t, 0, c] = torch.from_numpy(denormalized_channel).to(latents_history.device)

    # save image ================================
     # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    model_id = 'stabilityai/stable-diffusion-2-1'
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=inverse_scheduler,
        safety_checker=None,
        torch_dtype=dtype
    ).to(device)
    temp_dir = "output/temp_latents"
    os.makedirs(temp_dir, exist_ok=True)
    logging.info(f"Saving verifying intermediate results to {os.path.abspath(temp_dir)}...")

    step_size = max(len(latents_history) // 10, 1)
    for i in range(0, len(latents_history), step_size):
        latent = latents_history[i]

        # Save latent tensor
        torch.save(latent, os.path.join(temp_dir, f"latent_step_{i:04d}.pt"))

        # Visualize decoded image
        latent_viz = latent / 0.18215
        
        image = pipe.vae.decode(latent_viz).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        tvt.ToPILImage()(image[0]).save(
            os.path.join(temp_dir, f"latent_step_{i:04d}.png")
        )

        # Visualize latent channels
        latent_channels = latent_viz[0, :3]
        latent_channels = (latent_channels - latent_channels.min()) / (
            latent_channels.max() - latent_channels.min()
        )
        tvt.ToPILImage()(latent_channels).save(
            os.path.join(temp_dir, f"latent_channels_{i:04d}.png")
        )

    

    return result

if __name__ == "__main__":
    image_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/data/src_image_marco/elephant_resized.png"
    mask_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/output/Elephant_binary_mask_0.png"
    
    # Translation 30 pixels up
    transform_matrix = np.array([[1, 0, 10],
                               [0, 1, -10],
                               [0, 0, 1]], dtype=np.float32)
    
    try:
        result = test_modify_object(image_path, transform_matrix, mask_path)
        cv2.imwrite("result_64x64.png", result)
        print(f"Successfully saved result_64x64.png with shape {result.shape}")
    except Exception as e:
        print(f"Error: {e}")
