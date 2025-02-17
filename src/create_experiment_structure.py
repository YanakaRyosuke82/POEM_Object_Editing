import os
import shutil
import random
import cv2
import numpy as np


def create_folder_structure():
    # Create main folder
    base_dir = "exp_teaser_figure_auto_2"
    os.makedirs(os.path.join(base_dir, "input"), exist_ok=True)

    # Get list of prompt folders and their images
    generated_images_dir = "generated_images_2"
    prompt_folders = [f for f in os.listdir(generated_images_dir) if os.path.isdir(os.path.join(generated_images_dir, f)) and f.startswith("prompt")]

    if not prompt_folders:
        raise Exception("No prompt folders found in generated_images folder")

    # Create a list of all available images with their full paths
    available_images = []
    for prompt_folder in prompt_folders:
        prompt_path = os.path.join(generated_images_dir, prompt_folder)
        images = [f for f in os.listdir(prompt_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img in images:
            available_images.append(os.path.join(prompt_path, img))

    if not available_images:
        raise Exception("No images found in prompt folders")

    # Create a sample folder for each image
    for i, src_path in enumerate(available_images, 1):
        sample_dir = os.path.join(base_dir, "input", f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Copy the image
        dst_path = os.path.join(sample_dir, "input_image.png")
        shutil.copy2(src_path, dst_path)

        # Create mask with square at center
        img = cv2.imread(src_path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        square_size = min(h, w) // 3  # Make square 1/3 of image size
        x1 = (w - square_size) // 2
        y1 = (h - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        mask[y1:y2, x1:x2] = 255
        cv2.imwrite(os.path.join(sample_dir, "input_mask.png"), mask)

        # Create edit instruction
        edit_options = [
            "make the object twice as large",
            # "translate the object 50 pixels to the right",
            # "translate the object  75 pixels up",
            # "resize the object by 83%",
            # "resize the object by 132%",
        ]
        instruction = random.choice(edit_options)

        # Save edit instruction
        with open(os.path.join(sample_dir, "edit_instruction.txt"), "w") as f:
            f.write(instruction)

    print(f"Created {len(available_images)} samples successfully!")


if __name__ == "__main__":
    create_folder_structure()
