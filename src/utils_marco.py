import os
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
import logging
from PIL import Image


DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"

def generate_image_from_prompt(prompt: str = "a red apple on a white background", 
                                num_inference_steps: int = 50) -> torch.Tensor:
    """Generate image latents from a text prompt using Stable Diffusion.
    
    Args:
        prompt: Text prompt to generate image from
        num_inference_steps: Number of denoising steps
        
    Returns:
        Tensor of shape [num_steps, B, C, H, W] containing latents at each step
    """
    # Initialize Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Set scheduler to DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Store all intermediate latents during generation
    latents = None
    all_latents = []
    
    def callback(i, t, latents):
        all_latents.append(latents.clone())
    
    with torch.no_grad():
        latents = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            callback=callback,
            callback_steps=1,
            output_type="latent"
        ).images

    # Stack all latents into single tensor
    generated_latents = torch.stack(all_latents)  # Shape: [num_steps, B, C, H, W]
    
    logging.info(f"Generated latents shape: {generated_latents.shape}")
    
    return generated_latents


def get_mask_from_sam_detector(image: Image) -> Image:
    """
    Get a mask from the SAM detector.
    """
    
    return None


