import os
import logging
import torch
from PIL import Image
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import StableDiffusionPipeline_LEDITS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_id = 'runwayml/stable-diffusion-v1-5'
device = 'cuda'

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True

def load_model():
    """Load model and keep it in VRAM"""
    logger.info("Loading model...")
    pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model_id, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        model_id, subfolder="scheduler", algorithm_type="sde-dpmsolver++", solver_order=2
    )
    pipe.to(device)
    
    logger.info("Model loaded successfully!")
    return pipe

# Load model once globally
PIPELINE = load_model()

def process_image(input_image_path, instruction, output_path, steps=50, seed=1371, 
                 edit_guidance_scale=5.0, edit_threshold=0.75):
    """Process a single image using the model"""
    try:
        logger.info(f"Processing: {input_image_path}")
        
        # Set seed
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        
        # Invert the image directly from the input path
        logger.info(f"Inverting image: {input_image_path}")
        _ = PIPELINE.invert(image_path=input_image_path, num_inversion_steps=steps, skip=0.1)
        
        # Run inference
        logger.info(f"Applying edit with instruction: {instruction}")
        with torch.inference_mode():
            output = PIPELINE(
                editing_prompt=[instruction],
                edit_guidance_scale=edit_guidance_scale,
                edit_threshold=edit_threshold
                # Removed generator parameter as it's causing errors
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
                  edit_guidance_scale=5.0, edit_threshold=0.75):
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
                         edit_guidance_scale, edit_threshold))
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
    input_folder = "/work3/marscho/POSE_supplementary/onur_benchmark"
    process_folder(
        input_folder=input_folder,
        output_dir="onur_benchmark_output_ledits_plusplus",
        steps=50,
        seed=1371,
        edit_guidance_scale=7.5,
        edit_threshold=0.75
    )

if __name__ == "__main__":
    main()
