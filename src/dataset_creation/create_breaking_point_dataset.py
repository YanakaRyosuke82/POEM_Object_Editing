import os
import json
from PIL import Image, ImageDraw
import numpy as np
import shutil
import argparse


def get_unique_dataset_name(base_name):
    """Get a unique dataset name by adding a number suffix if needed"""
    if not os.path.exists(base_name):
        return base_name

    counter = 1
    while True:
        new_name = f"{base_name}_{counter}"
        if not os.path.exists(new_name):
            return new_name
        counter += 1


def create_mask(size=(512, 512), save_path="image_mask.png"):
    """Create a white square mask on black background"""
    # Create a black background
    mask = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(mask)

    # Calculate square dimensions (centered, 1/3 of the image size)
    square_size = min(size) // 3
    x1 = (size[0] - square_size) // 2
    y1 = (size[1] - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Draw white square
    draw.rectangle([x1, y1, x2, y2], fill="white")
    mask.save(save_path)
    return save_path


def create_breaking_point_dataset(image_path, dataset_name="breaking_point_dataset", object_name="mug"):
    """Create breaking point dataset with translation instructions"""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Get unique dataset name
    dataset_name = get_unique_dataset_name(dataset_name)

    # Create base directory structure
    base_dir = os.path.join(dataset_name, "input")
    os.makedirs(base_dir, exist_ok=True)

    # Generate samples for each translation in pixels
    for i, translation in enumerate(range(50, 401, 50), 1):
        # Create sample directory
        sample_dir = os.path.join(base_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Copy input image
        input_image_path = os.path.join(sample_dir, "input_image.png")
        shutil.copy2(image_path, input_image_path)

        # Create and save mask
        mask_path = os.path.join(sample_dir, "input_mask.png")
        create_mask(save_path=mask_path)

        # Create edit instruction file
        instruction_path = os.path.join(sample_dir, "edit_instruction.txt")
        with open(instruction_path, "w") as f:
            f.write(f"move the {object_name} to the left by {translation} pixels")

    print(f"Dataset created with {i} samples in {dataset_name}/input/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a breaking point dataset with translation instructions")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--dataset-name", type=str, default="breaking_point_dataset", help="Name of the dataset (default: breaking_point_dataset)")
    parser.add_argument("--name", type=str, default="mug", help="Name of the object to translate (default: mug)")

    args = parser.parse_args()

    create_breaking_point_dataset(args.image_path, args.dataset_name, args.name)
