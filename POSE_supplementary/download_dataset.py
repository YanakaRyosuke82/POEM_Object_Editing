from datasets import load_dataset
import os
import argparse


args = argparse.ArgumentParser()
args.add_argument("--in_dir", type=str, required=True, help="Input images directory")
args = args.parse_args()

os.makedirs(args.in_dir, exist_ok=True)

dataset = load_dataset("monurcan/precise_benchmark_for_object_level_image_editing", split="train")
dataset = dataset.to_iterable_dataset()
# dataset = dataset.take(args.dataset_size_samples)

from tqdm import tqdm

for sample in tqdm(dataset, desc="Processing samples"):
    input_image, user_promppt, save_file_name = sample["input_image"], sample["edit_prompt"], sample["id"]
    input_mask = sample["input_mask"]

    subfolder_name = save_file_name
    # Create input subfolder for this sample
    sample_input_dir = os.path.join(args.in_dir, subfolder_name)
    os.makedirs(sample_input_dir, exist_ok=True)
    input_path = os.path.join(args.in_dir, subfolder_name, "input_image.png")
    edit_instruction_file = os.path.join(args.in_dir, subfolder_name, "edit_instruction.txt")
    with open(edit_instruction_file, "w") as file:
        file.write(user_promppt)
    input_image.save(input_path)
    input_mask.save(os.path.join(args.in_dir, subfolder_name, "input_mask.png"))
    # save the savefilename to a txt file
    with open(os.path.join(args.in_dir, subfolder_name, "save_file_name.txt"), "w") as file:
        file.write(save_file_name)
