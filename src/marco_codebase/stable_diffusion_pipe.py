import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np

import pdb
# pdb.set_trace()

def run_stable_diffusion(
    prompt: str,
    num_inference_steps: int = 500,
    guidance_scale: float = 7.5,
    negative_prompt: str = None,
    height: int = 512,
    width: int = 512,
    generator: torch.Generator = None
) -> Image.Image:
    """
    Run standard stable diffusion pipeline with all components.
    
    Args:
        prompt: Text prompt to condition the generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Scale for classifier-free guidance
        negative_prompt: Optional negative prompt
        height: Output image height 
        width: Output image width
        generator: Optional random generator for reproducibility
        
    Returns:
        PIL Image generated from the prompt
    """
    # Initialize pipeline
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Set scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps)  # Ensure timesteps are set

    # Enable memory efficient attention
    pipe.enable_attention_slicing()
    
    # Generate initial random latents
    latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device="cuda",
        dtype=torch.float16
    )
    
    # Scale latents as they are in [-1, 1] but model expects [-0.18215, 0.18215]
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # Prepare text embeddings
    text_input = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to("cuda"))[0]
    
    # Prepare negative prompt embeddings if provided
    if negative_prompt is None:
        negative_prompt = ""
    uncond_input = pipe.tokenizer(
        [negative_prompt],
        padding="max_length", 
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to("cuda"))[0]
    
    # Concatenate conditional and unconditional embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    # Load modified latents history
    modified_latents_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/output/modified_latents/modified_latents_history.pt"
    modified_latents = torch.load(modified_latents_path).to("cuda")
    modified_latents = modified_latents.flip(0)  # Reverse the tensor on the first dimension
    modified_latents_iter = iter(modified_latents)  # Initialize the iterator for modified_latent

    # Load and prepare mask1
    mask1_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/output/Elephant_binary_mask_0.png"
    mask1 = Image.open(mask1_path).convert('L')
    mask1 = mask1.resize((height // 8, width // 8))
    mask1 = torch.from_numpy(np.array(mask1)).to(device=latents.device, dtype=latents.dtype) / 255.0
    mask1 = mask1.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Create a binary mask that looks like a rectangle based on the loaded mask
    non_zero_indices = torch.nonzero(mask1)
    min_y, min_x = non_zero_indices[:, 2].min(), non_zero_indices[:, 3].min()
    max_y, max_x = non_zero_indices[:, 2].max(), non_zero_indices[:, 3].max()

    # Create a rectangular mask (bounding box) and fill it
    mask1[:, :, min_y:max_y + 1, min_x:max_x + 1] = 1.0

    # Load and prepare mask2
    mask2_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/data/elephant_transformed_mask.png"
    mask2 = Image.open(mask2_path).convert('L')
    mask2 = mask2.resize((height // 8, width // 8))
    mask2 = torch.from_numpy(np.array(mask2)).to(device=latents.device, dtype=latents.dtype) / 255.0
    mask2 = mask2.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Calculate the difference between mask1 and mask2
    mask_difference = (mask1 - mask2).clamp(0, 1)  # Ensure no negative values

    # Convert mask_difference to a binary mask
    binary_mask_difference = (mask_difference > 0.5).float()

    # Save binary mask_difference as an image to disk
    binary_mask_difference_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/output/binary_mask_difference.png"
    binary_mask_difference_image = Image.fromarray((binary_mask_difference.squeeze().cpu().numpy() * 255).astype(np.uint8))
    binary_mask_difference_image.save(binary_mask_difference_path)
    print(f"Binary mask difference saved as an image to {binary_mask_difference_path}")

    from tqdm import tqdm

    # Run the denoising loop
    num_iterations = len(pipe.scheduler.timesteps)
    for step_index, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="Denoising")):
        # Prepare latents for classifier-free guidance by duplicating
        latent_model_input = torch.cat([latents] * 2)
        
        # Adjust latents based on the current timestep
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict the noise residual using the UNet model
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            
        # Apply guidance to the predicted noise
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Update latents to the previous noisy sample
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Integrate mask difference into latents only for the first 20% of iterations
        if step_index < 0.85 * num_iterations:
            modified_latent = next(modified_latents_iter)
            # Take weighted average based on mask difference
            latents = latents * mask_difference + modified_latent * (1 - mask_difference)
        
        # Convert latents to an image for visualization and save
        latents_image = (latents / 0.18215).clamp(-1, 1)
        latents_image = pipe.vae.decode(latents_image).sample
        latents_image = (latents_image / 2 + 0.5).clamp(0, 1)
        latents_image = latents_image.detach().cpu().permute(0, 2, 3, 1).numpy()
        latents_image = (latents_image * 255).round().astype("uint8")
        latents_image = Image.fromarray(latents_image[0])
        latents_image.save(f"/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/output/latents_step_{step_index:04d}.png")

        
        
     
    # Decode latents to image
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
        
    # Convert to PIL image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])
    
    return image


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A realistic photo of an elephant")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=500)
    args = parser.parse_args()

    # Generate image
    image = run_stable_diffusion(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps
    )

    # Save output
    image.save(args.output)
    print(f"Generated image saved to {args.output}")
