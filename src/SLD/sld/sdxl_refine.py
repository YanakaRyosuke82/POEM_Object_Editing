from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
import os
import time

def sdxl_refine(prompt, input_fname, output_fname, evaluation_folder, save_file_name):
    torch.set_float32_matmul_precision("high")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
    )

    pipe = pipe.to("cuda")

    init_image = Image.open(input_fname)
    init_image = init_image.resize((1024, 1024), Image.LANCZOS)
    image = pipe(
        prompt,
        image=init_image,
        strength=0.3,
        aesthetic_score=7.0,
        num_inference_steps=50,
    ).images
    image[0].save(output_fname)

    # Save the image to the evaluation folder
    image[0].save(os.path.join(evaluation_folder, f"{save_file_name}.png"))