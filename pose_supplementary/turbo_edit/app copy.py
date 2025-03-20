from __future__ import annotations

import os
import argparse
from PIL import Image
import torch

from my_run import run as run_model


def main():
    parser = argparse.ArgumentParser(description="Turbo Edit")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--src_prompt", type=str, required=True, help="Source prompt")
    parser.add_argument("--tgt_prompt", type=str, required=True, help="Target prompt")
    parser.add_argument("--seed", type=int, default=7865, help="Random seed")
    parser.add_argument("--w1", type=float, default=1.5, help="Weight parameter")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")

    args = parser.parse_args()

    # Fixed w2 value as in the original code
    w2 = 1.0

    print(f"Processing image: {args.input_image}")
    print(f"Source prompt: {args.src_prompt}")
    print(f"Target prompt: {args.tgt_prompt}")
    print(f"Using seed: {args.seed}, w1: {args.w1}, w2: {w2}")

    # Run the model
    result_image = run_model(
        args.input_image,
        args.src_prompt,
        args.tgt_prompt,
        args.seed,
        args.w1,
        w2
    )

    # Save the result
    result_image.save(args.output)
    print(f"Result saved to: {args.output}")

    # Example command for easy copy-paste:
    # python app.py --input_image examples_demo/1.jpeg --src_prompt "a dreamy cat sleeping on a floating leaf" --tgt_prompt "a dreamy bear sleeping on a floating leaf" --seed 7 --w1 1.3 --output output1.png

    print("\nExample usage:")
    print("python app.py --input_image examples_demo/1.jpeg --src_prompt \"a dreamy cat sleeping on a floating leaf\" --tgt_prompt \"a dreamy bear sleeping on a floating leaf\" --seed 7 --w1 1.3 --output output1.png")
    print("python app.py --input_image examples_demo/2.jpeg --src_prompt \"A painting of a cat and a bunny surrounded by flowers\" --tgt_prompt \"a polygonal illustration of a cat and a bunny\" --seed 2 --w1 1.5 --output output2.png")
    print("python app.py --input_image examples_demo/3.jpg --src_prompt \"a chess pawn wearing a crown\" --tgt_prompt \"a chess pawn wearing a hat\" --seed 2 --w1 1.3 --output output3.png")

if __name__ == "__main__":
    main()
