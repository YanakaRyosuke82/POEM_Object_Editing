"""

https://huggingface.co/learn/diffusion-course/en/unit4/2#loading-an-existing-pipeline
https://github.com/shaibagon/diffusers_ddim_inversion/blob/main/ddim_inversion.py




DDIM Inversion for Stable Diffusion.

This module implements DDIM inversion for Stable Diffusion models, allowing reconstruction
of latent representations from input images. It includes functionality for:
- Loading and preprocessing images
- Converting images to latent space
- Running DDIM inversion
- Visualizing and saving intermediate results

The main function ddim_inversion() performs the inversion process and optionally verifies
the results by reconstructing the original image.
"""

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
from torchvision import transforms as tvt


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    """
    Load and preprocess an image file.

    Args:
        imgname: Path to the image file
        target_size: Optional target size as int (square) or (height, width) tuple

    Returns:
        Preprocessed image tensor with batch dimension [1, C, H, W]
    """
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """
    Convert image tensor to latent representation using VAE.

    Args:
        x: Image tensor [B, C, H, W]
        vae: Pretrained VAE model

    Returns:
        Latent representation scaled by 0.18215
    """
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    return posterior.mean * 0.18215


@torch.no_grad()
def ddim_inversion(imgname: str, num_steps: int = 50, verify: bool = False) -> torch.Tensor:
    """
    Perform DDIM inversion on an input image.

    Args:
        imgname: Path to input image
        num_steps: Number of denoising steps
        verify: If True, verify reconstruction quality

    Returns:
        Tensor containing latent representations at each step
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Running DDIM inversion on {imgname}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # Initialize models
    model_id = 'stabilityai/stable-diffusion-2-1'
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=inverse_scheduler,
        safety_checker=None,
        torch_dtype=dtype
    ).to(device)
    
    # Process input image
    input_img = load_image(imgname).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, pipe.vae)

    # Run inversion
    latents_history = []
    def callback_on_step_end(pipe, step, timestep, callback_kwargs):
        latents_history.append(callback_kwargs["latents"].detach().clone())
        return callback_kwargs

    inv_latents, _ = pipe(
        prompt="A realistic photo of an elephant",
        negative_prompt="",
        guidance_scale=1.0,
        width=input_img.shape[-1],
        height=input_img.shape[-2],
        output_type='latent',
        return_dict=False,
        num_inference_steps=num_steps,
        latents=latents,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=["latents"]
    )

    latents_history = torch.stack(latents_history)

    # Optional verification
    if verify:
        logging.info("Verifying inversion results...")
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
        image = pipe(
            prompt="A realistic photo of an elephant",
            negative_prompt="",
            guidance_scale=1.0,
            num_inference_steps=num_steps,
            latents=inv_latents
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(tvt.ToPILImage()(input_img[0]))
        ax2.imshow(image.images[0])
        plt.savefig('output/verification_result.png')

        # Save intermediate results
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


    return latents_history


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ddim_inversion(
        '/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/data/src_image_marco/elephant_resized.png',
        num_steps=100,
        verify=True
    )
