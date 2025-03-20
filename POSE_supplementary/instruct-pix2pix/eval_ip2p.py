import os
import logging
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_id = "timbrooks/instruct-pix2pix"

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True

def load_model():
    """Load model and keep it in VRAM"""
    logger.info("Loading model...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision for better memory usage
        safety_checker=None
    ).to("cuda")

    # Enable memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Using xformers for optimization")
    except Exception as e:
        logger.warning(f"xformers not available: {e}")

    # Optimize memory layout
    pipe.unet.to(memory_format=torch.channels_last)
    
    logger.info("Model loaded successfully!")
    return pipe

# Load model once globally
PIPELINE = load_model()
def process_image(input_image_path, instruction, output_path, steps=50, seed=1371, 
                 text_cfg_scale=7.5, image_cfg_scale=1.5):
    """Process a single image using the model"""
    try:
        logger.info(f"Processing: {input_image_path}")

        # Load and resize image
        input_image = Image.open(input_image_path).convert("RGB")

        # Resize to multiple of 64 while maintaining aspect ratio
        width, height = input_image.size
        scale_factor = 512 / max(width, height)
        new_width = max(64, int((width * scale_factor) // 64) * 64)
        new_height = max(64, int((height * scale_factor) // 64) * 64)

        input_image = ImageOps.fit(input_image, (new_width, new_height), method=Image.Resampling.LANCZOS)

        # Validate image dimensions
        if min(input_image.size) < 64:
            logger.error(f"Image too small after resizing: {input_image.size}")
            return False

        # Ensure steps are within model capacity
        steps = min(steps, 50)  # Clamp to 50 (default upper limit for StableDiffusion)

        # Run inference
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                generator = torch.manual_seed(seed)

                # Convert image to tensor (ensuring proper shape)
                output = PIPELINE(
                    instruction,
                    image=input_image,
                    guidance_scale=text_cfg_scale,
                    image_guidance_scale=image_cfg_scale,
                    num_inference_steps=steps,
                    generator=generator,
                )

                if not output.images:
                    logger.error(f"Model did not return an image for {input_image_path}")
                    return False

                edited_image = output.images[0]

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        edited_image.save(output_path)
        logger.info(f"Saved edited image to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {input_image_path}: {e}")
        return False


def process_folder(input_folder, output_dir="output", steps=50, seed=1371, 
                  text_cfg_scale=7.5, image_cfg_scale=1.5):
    """Process all images in a folder sequentially for better reliability"""
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    # Collect all valid tasks
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
                
            tasks.append((input_image_path, instruction, output_path, steps, seed, 
                         text_cfg_scale, image_cfg_scale))
        except Exception as e:
            logger.error(f"Error reading instruction from {instruction_file}: {e}")

    # Process tasks
    if not tasks:
        logger.info("No tasks to process")
        return

    # Process images sequentially instead of using multithreading
    logger.info(f"Processing {len(tasks)} images sequentially")
    
    successful_count = 0
    for task in tasks:
        if process_image(*task):
            successful_count += 1
    
    logger.info(f"Completed processing {successful_count} of {len(tasks)} images successfully")

def main():
    input_folder = "/work3/marscho/POSE_supplementary/onur_benchmark"
    process_folder(
        input_folder=input_folder,
        output_dir="onur_benchmark_output",
        steps=50,
        seed=1371,
        text_cfg_scale=7.5,
        image_cfg_scale=1.5
    )

if __name__ == "__main__":
    main()
