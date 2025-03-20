import logging
import os
import re
from typing import Dict, Any, List
import numpy as np
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
import torch


def parse_detection_file(file_path):
    """Parse detection file and extract object information."""
    with open(file_path, "r") as f:
        detection_data = f.read()

    # Extract scene description and spatial relationships
    scene_desc_match = re.search(r"Scene Description:\n(.*?)\n", detection_data)
    spatial_rel_match = re.search(r"Spatial Relationships:\n(.*?)\n", detection_data)

    scene_desc = scene_desc_match.group(1) if scene_desc_match else ""
    spatial_rel = spatial_rel_match.group(1) if spatial_rel_match else ""

    objects = []
    pattern = r"Object (\d+):\n\s+Class: (.*?)\n\s+Bounding Box.*?xmin=([\d.]+), ymin=([\d.]+), xmax=([\d.]+), ymax=([\d.]+)"

    for match in re.finditer(pattern, detection_data):
        xmin, ymin, xmax, ymax = map(float, match.groups()[2:])

        corners = np.array([[xmin, ymin, 1], [xmax, ymin, 1], [xmax, ymax, 1], [xmin, ymax, 1]])  # top-left  # top-right  # bottom-right  # bottom-left

        obj = {
            "id": int(match.group(1)),
            "class": match.group(2),
            "bbox": [xmin, ymin, xmax, ymax],
            "corners": corners,
            "width": xmax - xmin,
            "height": ymax - ymin,
            "center": [(xmax + xmin) / 2, (ymax + ymin) / 2],
        }
        objects.append(obj)

    return objects, scene_desc, spatial_rel


import numpy as np
import re


def parse_transformation_matrix_vlm(reasoning):
    """Extract and parse transformation matrix from model reasoning."""
    # Find the start and end tokens
    start_token = "<<MATRIX_START>>"
    end_token = "<<MATRIX_END>>"

    start_idx = reasoning.find(start_token)
    end_idx = reasoning.find(end_token, start_idx)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find the transformation matrix in the reasoning output.")

    # Extract the text between the tokens
    matrix_text = reasoning[start_idx + len(start_token) : end_idx].strip()

    # Use regex to extract numbers, handling both integer and float formats
    pattern = r"[-+]?(?:\d*\.\d+|\d+\.?)"
    numbers = re.findall(pattern, matrix_text)

    if len(numbers) < 9:
        raise ValueError(f"Could not find 9 numbers in the matrix text. Found: {len(numbers)}")

    try:
        # Convert strings to floats and reshape into 3x3 matrix
        # Ensure float format by adding .0 to integers
        matrix_values = []
        for n in numbers[:9]:
            if "." not in n:
                n = n + ".0"
            matrix_values.append(float(n))

        matrix = np.array(matrix_values, dtype=float).reshape(3, 3)
    except Exception as e:
        raise ValueError(f"Error parsing the matrix: {e}")

    # Validate the matrix shape
    if matrix.shape != (3, 3):
        raise ValueError("The parsed matrix is not a 3x3 matrix.")

    return matrix


def parse_object_id_vlm(reasoning):
    """Extract object ID from VLM model reasoning."""
    # Find object ID between start and end tags
    id_start = reasoning.find("<<OBJECT_ID_START>>")
    id_end = reasoning.find("<<OBJECT_ID_END>>")

    if id_start == -1 or id_end == -1:
        raise ValueError("Could not find object ID tags in VLM output")

    try:
        object_id = int(reasoning[id_start + len("<<OBJECT_ID_START>>") : id_end])
        return object_id
    except:
        raise ValueError("Failed to parse object ID from VLM output")


def generate_prompt(model_name, user_edit, objects, scene_desc, spatial_rel, device):
    scene_context = f"""Scene Information:
    
    Spatial Relationships:
    {spatial_rel}
    
    OBJECT DETAILS:"""

    for obj in objects:
        scene_context += f"""
        Object ID       : {obj['id']}
        Object class    : {obj['class']}
        Width          : {obj['width']:.3f}
        Height         : {obj['height']:.3f}
        Center         : ({obj['center'][0]:.3f}, {obj['center'][1]:.3f})
        ----------------------"""

    messages_qwen_vlm = [
        {
            "role": "system",
            "content": (
                "You are a computer vision math expert. Your task is to output a 3x3 transformation matrix and object ID based on the user's edit request.\n\n"
                "OUTPUT FORMAT REQUIREMENTS:\n"
                "1. Object ID in format: <<OBJECT_ID_START>>N<<OBJECT_ID_END>> where N is the integer ID\n"
                "2. Matrix in format:\n"
                "<<MATRIX_START>>\n"
                "[[a.aa  b.bb  c.cc]\n"
                " [d.dd  e.ee  f.ff]\n"
                " [g.gg  h.hh  i.ii]]\n"
                "<<MATRIX_END>>\n\n"
                "MATRIX RULES:\n"
                "- Must be exactly 3x3\n"
                "- All numbers must be floats with 2 decimal places (1.00 not 1)\n"
                "- Use exactly 2 spaces between numbers\n"
                "- Image coordinates: origin at top-left, X right, Y down\n\n"
                "AVAILABLE TRANSFORMATIONS templates:\n"
                "Translation: [[1.00  0.00  tx], [0.00  1.00  ty], [0.00  0.00  1.00]]\n"
                "Rotation: [[cos(θ)  -sin(θ)  0.00], [sin(θ)  cos(θ)  0.00], [0.00  0.00  1.00]]\n"
                "Scale: [[sx  0.00  0.00], [0.00  sy  0.00], [0.00  0.00  1.00]]\n"
                "Shear: [[1.00  shx  0.00], [shy  1.00  0.00], [0.00  0.00  1.00]]\n"
                "Flip X: [[-1.00  0.00  0.00], [0.00  1.00  0.00], [0.00  0.00  1.00]]\n\n"
                "COMBINATIONS:\n"
                "- Translate then rotate: multiply matrices in order T*R\n"
                "- Scale then translate: multiply matrices in order T*S\n"
                "- Rotate then scale: multiply matrices in order S*R\n"
                "- Any sequence: multiply matrices from right to left\n"
                "Example: Translate(tx,ty) * Rotate(θ) * Scale(sx,sy)\n\n"
                "SCENE CONTEXT:\n"
                f"{scene_context}\n\n"
                "CRITICAL: Your response must contain exactly one matrix between <<MATRIX_START>> and <<MATRIX_END>> tokens, "
                "and exactly one object ID between <<OBJECT_ID_START>> and <<OBJECT_ID_END>> tokens.\n"
                "The object ID must be one of the IDs listed in OBJECT DETAILS."
            ),
        },
        {"role": "user", "content": user_edit},
    ]

    return messages_qwen_vlm


def run_math_llm(model_name, model, processor, user_edit, objects, scene_desc, spatial_rel, device, logger):

    # 1. generate prompt
    messages = generate_prompt(model_name, user_edit, objects, scene_desc, spatial_rel, device)
    # Try parsing up to 5 times
    for attempt in range(5):

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Generate output
        with torch.cuda.device(device):
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            output_text = processor.batch_decode(
                [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        reasoning = output_text

        logger.info(f"Model Reasoning (Attempt {attempt + 1}):")
        logger.info("Marco Reasoning: " + reasoning)

        try:
            if model_name == "qwen_vlm":
                matrix = parse_transformation_matrix_vlm(reasoning)
                object_id = parse_object_id_vlm(reasoning)
            else:
                raise ValueError(f"Model {model_name} not supported")
            return matrix, object_id, reasoning
        except Exception as e:  # Catch specific exception for better error handling
            if attempt == 4:  # Last attempt
                logger.warning(f"Failed to parse matrix after 5 attempts: {str(e)}, using identity matrix and object_id=1")
                identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                return identity_matrix, 1, reasoning
            logger.error(f"Failed to parse matrix on attempt {attempt + 1}: {str(e)}, retrying...")
            continue


def run_math_analysis_vlm(user_edit: str, file_path: str, model_name: str, model: Any, tokenizer: Any, device: str, logger):
    try:
        # 0. parse detection file with enhanced information
        file_dir = os.path.dirname(file_path)
        objects, scene_desc, spatial_rel = parse_detection_file(file_path)

        # 2. run math llm
        matrix_array, object_id, reasoning = run_math_llm(model_name, model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device, logger)

        logger.info("Parsed Matrix: \n" + str(matrix_array))
        logger.info("Object ID:" + str(object_id))

        # 3. save files
        TRANSFORMATION_MATRIX_FILE = f"{file_dir}/transformation_matrix.npy"
        np.save(TRANSFORMATION_MATRIX_FILE, matrix_array)

        REASONING_FILE = f"{file_dir}/math_reasoning.txt"
        with open(REASONING_FILE, "w") as f:
            f.write(reasoning)

        OBJECT_ID_FILE = f"{file_dir}/object_id.txt"
        with open(OBJECT_ID_FILE, "w") as f:
            f.write(str(object_id))

        return matrix_array, object_id

    except Exception as e:
        logging.error(f"Error in mathematical analysis for {file_path}: {str(e)}")
        raise
