source_prompt = "a bowl of fruits"
target_prompt = "a bowl of pears"

import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

from diffusers.utils import load_image, make_image_grid

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))


from PIL import Image

source_prompt = "a bowl of fruits"
target_prompt = "a basket of pears"
mask_image = pipeline.generate_mask(
    image=raw_image,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
)
Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))

inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents

output_image = pipeline(
    prompt=target_prompt,
    mask_image=mask_image,
    image_latents=inv_latents,
    negative_prompt=source_prompt,
).images[0]
mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))


grid_image = make_image_grid([raw_image, mask_image, output_image], rows=1, cols=3)
grid_image.save("output_image_grid.png")
