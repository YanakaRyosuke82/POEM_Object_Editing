import os
import shutil
import cv2


def main():
    # Define input and output paths
    input_dir = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/benchmark/exp2/output_grounding_dino_vlm_qwen_math_qwen"
    output_dir = "/dtu/blackhole/00/215456/marcoshare/FINAL_FINAL_LAST_RESULTS_SUNDAY/5_qwen_MLLM_grounded_SAM"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all subdirectories
    for subdir, _, files in os.walk(input_dir):
        # Check if mask_1.png exists in this directory
        if "mask_1.png" not in files:
            continue

        # Try to read object_id.txt
        object_id_path = os.path.join(subdir, "object_id.txt")
        if not os.path.exists(object_id_path):
            continue

        try:
            with open(object_id_path, "r") as f:
                object_id = int(f.read().strip())
        except (ValueError, IOError):
            continue

        # Check if mask_{object_id}.png exists
        mask_filename = f"mask_{object_id}.png"
        mask_path = os.path.join(subdir, mask_filename)
        if not os.path.exists(mask_path):
            continue

        # Get a unique name for the output file using the subfolder name
        subfolder_name = os.path.basename(subdir)
        output_filename = f"{subfolder_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Copy the mask file to the output directory
        shutil.copy2(mask_path, output_path)
        print(f"Copied {mask_path} to {output_path}")


if __name__ == "__main__":
    main()
