from diffusers import DiffusionPipeline
import torch
import sys

# load both base & refiner
base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 50
high_noise_frac = 0.8

# Get dataset_name as argument
if len(sys.argv) < 2:
    print("Usage: python generate_synthetic_image.py <dataset_name>")
    sys.exit(1)

dataset_name = "synthetic_dataset_" + sys.argv[1]

# prompts = [
#     "A realistic image of one single red apple and an orange pumpkin on a dark brown wooden table against a deep black background. Soft, moody lighting highlights their textures, with the apple smooth and shiny, and the pumpkin rough with visible ridges."
# ]
prompts = [
    "a photorealistic image of a single ball on a white surface with an Icelandic landscape in the background",
    "a photorealistic image of a gold coin on a plain background with Danish countryside scenery",
    "a photorealistic image of a green pear on a light gray background with Icelandic mountains in the distance",
    "a photorealistic image of a simple wooden bowl on a plain background with a Danish forest setting",
    "a photorealistic image of a single orange on a white surface with an Icelandic beach in the background",
]
# prompts = [
#     "a green tennis ball and a blue toy car on a white surface, both objects clearly visible and easily movable, soft even lighting from above, minimalist composition with solid neutral background, photographic quality"
# ]


# prompts = [
#     "an orange tabby cat sitting next to a big red ball on a light gray floor, both objects clearly visible and centered in frame, soft even lighting from above, minimalist composition with solid neutral background, photographic quality",
#     "a blue mug next to a red apple on a matte brown wooden table, objects placed side by side with small gap between them, soft diffused natural lighting from above, centered in frame, clean simple composition with solid light gray background, photographic quality",
# ]

# prompts = [
#     "a red apple and a yellow banana lying side by side on a light gray table, soft natural lighting, objects fully visible in frame, simple composition",
#     "a blue coffee mug next to a white plate on a light beige table, soft ambient lighting, objects centered and fully visible, minimal setting",
#     "a green tennis ball next to a black tv remote on a pale blue desk, gentle overhead lighting, objects clearly visible and centered, simple setup",
#     "a brown leather wallet beside a silver smartphone on a light cream surface, soft diffused lighting, objects positioned centrally, clean composition",
#     "a red pencil next to a blue eraser on a soft tan desk surface, even lighting, objects placed centrally in frame, minimalist scene",
# ]


import os
from utils_pose.crop_image import resize_and_crop_image

for i, prompt in enumerate(prompts):
    # Create folder for this prompt
    prompt_dir = f"{dataset_name}/prompt_{i+1}"
    os.makedirs(prompt_dir, exist_ok=True)

    # Generate 10 images for this prompt
    for j in range(10):
        # run both experts
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        # Save original image
        image_path = os.path.join(prompt_dir, f"image_{j+1}.png")
        image.save(image_path)

        # Resize and crop the generated image to 512x512
        resize_and_crop_image(image_path, image_path, target_size=(512, 512))
