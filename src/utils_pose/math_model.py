import logging
import os
import re
from typing import Dict, Any, List
import numpy as np
import torch.nn.functional as F


# Define transformation matrices with clear mathematical formulas
class TransformationMatrices:
    @staticmethod
    def translation(tx, ty):
        return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    @staticmethod
    def rotation(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    @staticmethod
    def scaling(sx, sy):
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    @staticmethod
    def shear(shx, shy):
        return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])


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


def parse_transformation_matrix_qwen(reasoning):
    """Extract and parse transformation matrix from model reasoning."""
    # Look for matrix pattern in the output format, including LaTeX style matrices
    # Find the output marker first
    output_start = reasoning.find("output")
    if output_start == -1:
        raise ValueError("Could not find output section in model reasoning")

    # Search for matrix patterns after the output marker
    reasoning_after_output = reasoning[output_start:]

    matrix_patterns = [
        # Standard array format
        r"\[\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
        r"\s*,\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
        r"\s*,\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*\]",
        # LaTeX style matrix pattern with pmatrix
        r"\\begin{pmatrix}\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*\\\\"
        r"\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*\\\\"
        r"\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*\\end{pmatrix}",
        # LaTeX style matrix pattern with brackets
        r"\\\[\s*\\begin{pmatrix}\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*\\\\"
        r"\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*\\\\"
        r"\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*&\s*([-+]?\d*\.?\d+)\s*\\end{pmatrix}\s*\\\]",
        # Standard array format with integers
        r"\[\s*\[\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\]"
        r"\s*,\s*\[\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\]"
        r"\s*,\s*\[\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\]\s*\]",
        # LaTeX style matrix pattern with integers
        r"\\begin{pmatrix}\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*\\\\"
        r"\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*\\\\"
        r"\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*\\end{pmatrix}",
        # LaTeX style matrix pattern with brackets and integers
        r"\\\[\s*\\begin{pmatrix}\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*\\\\"
        r"\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*\\\\"
        r"\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*&\s*([-+]?\d+)\s*\\end{pmatrix}\s*\\\]",
    ]

    matrix_values = None
    for pattern in matrix_patterns:
        match = re.search(pattern, reasoning_after_output)
        if match:
            matrix_values = match.groups()
            break

    if not matrix_values:
        raise ValueError("Could not find transformation matrix in model output section")

    # Convert all values to floats
    matrix = np.array([float(val) for val in matrix_values]).reshape(3, 3)

    return matrix


def parse_transformation_matrix_deepseek(reasoning):
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


def parse_appearance_token_qwen(reasoning):
    """Extract and parse appearance token from model reasoning."""
    pattern = r"APPEARANCE\s*:\s*([a-zA-Z]+)|APpearances\s*:\s*([a-zA-Z]+)"
    match = re.search(pattern, reasoning, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    else:
        raise ValueError("Could not find appearance token in the reasoning output.")


def parse_object_id_qwen(reasoning):
    """Extract and parse object ID (integer) from model reasoning."""
    # Try different patterns for object ID
    patterns = [
        # Standard format
        r"OBJECT_ID:\s*(\d+)",
        # LaTeX boxed format
        r"\\boxed{(\d+)}",
        # Plain number after comma in LaTeX
        r"\\end{pmatrix},\s*(\d+)",
        # Number after comma
        r"matrix\},\s*(\d+)",
        # Just the number after OBJECT_ID:
        r"OBJECT_ID: (\d+)",
        # Object ID in code block
        r"object_id = (\d+)",
        # Object ID in output block
        r"OBJECT_ID: (\d+)",
        # Object ID in LaTeX equation
        r"\\text{Object ID} = (\d+)",
        # Object ID in natural language
        r"object (?:ID|id|Id) (?:is|=) (\d+)",
        # Object ID in parentheses
        r"\(object (?:ID|id|Id):\s*(\d+)\)",
        # Object ID after colon
        r"(?:ID|id|Id):\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                object_id = int(match.group(1))
                return object_id
            except ValueError:
                continue

    # If no pattern matched, try to find just the number after "OBJECT_ID:"
    if "OBJECT_ID:" in reasoning:
        try:
            id_text = reasoning.split("OBJECT_ID:")[1].strip().split()[0]
            return int(id_text)
        except (IndexError, ValueError):
            pass

    # Try to find any standalone number in the text
    numbers = re.findall(r"\b\d+\b", reasoning)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            pass

    raise ValueError("Could not find valid object ID in the reasoning output.")


def parse_object_id_deepseek(reasoning):
    """Extract and parse object ID (integer) from model reasoning."""
    # Find the start and end tokens
    start_token = "<<OBJECT_ID_START>>"
    end_token = "<<OBJECT_ID_END>>"

    start_idx = reasoning.find(start_token)
    end_idx = reasoning.find(end_token, start_idx)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find the object ID in the reasoning output.")

    # Extract text between tokens and parse as integer
    id_text = reasoning[start_idx + len(start_token) : end_idx].strip()
    try:
        # Remove "Object #" prefix if present and convert to int
        id_text = id_text.replace("Object #", "").strip()
        object_id = int(id_text)
        return object_id
    except ValueError:
        raise ValueError(f"Could not parse object ID from text: {id_text}")


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

    messages_deepseek_r1_distill_qwen_32B_ONUR = [
        {
            "role": "system",
            "content": (
                "You are a computer vision math expert. Your task is to output a 3x3 transformation matrix and object ID based on the user's edit request.\n\n"
                "OUTPUT FORMAT REQUIREMENTS:\n"
                "1. Object ID in format: <<OBJECT_ID_START>>N<<OBJECT_ID_END>> where N is the integer ID\n"
                "2. Matrix in format:\n"
                "3. Appearance description in format: <<APPEARANCE_START>><object_id>|<object_class>|<appearance_description><<APPEARANCE_END>>\n"
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
                "SCENE CONTEXT:\n"
                f"{scene_context}\n\n"
                "CRITICAL: Your response must contain exactly one matrix between <<MATRIX_START>> and <<MATRIX_END>> tokens, "
                "and exactly one object ID between <<OBJECT_ID_START>> and <<OBJECT_ID_END>> tokens.\n"
                "The object ID must be one of the IDs listed in OBJECT DETAILS."
            ),
        },
        {"role": "user", "content": user_edit},
    ]
    messages_qwen2_5_math_7b_instruct = [
        {
            "role": "system",
            "content": "Please integrate natural language reasoning with programs to solve the problem. Your task is to output a 3x3 transformation matrix and object ID based on the user's edit request.\n\n"
            "Scene content: " + scene_context + "\n\n"
            "REQUIRED OUTPUT:\n"
            "1. The word 'MATRIX' followed by the 3x3 transformation matrix\n"
            "2. The word 'OBJECT_ID' followed by the object ID number\n"
            "3. The word 'APPEARANCE' followed by the appearance token\n\n"
            "TRANSFORMATION MATRIX TEMPLATES:\n"
            "Translation: [[1.00  0.00  tx], [0.00  1.00  ty], [0.00  0.00  1.00]]\n"
            "Rotation: [[cos(θ)  -sin(θ)  0.00], [sin(θ)  cos(θ)  0.00], [0.00  0.00  1.00]]\n"
            "Scale: [[sx  0.00  0.00], [0.00  sy  0.00], [0.00  0.00  1.00]]\n"
            "Shear: [[1.00  shx  0.00], [shy  1.00  0.00], [0.00  0.00  1.00]]\n"
            "Flip X: [[-1.00  0.00  0.00], [0.00  1.00  0.00], [0.00  0.00  1.00]]\n\n"
            "Combined transformations: Multiply matrices with right order.\n"
            "For example, to scale then translate: Translation_matrix @ Scale_matrix\n\n"
            "MATRIX RULES:\n"
            "- Must be exactly 3x3\n"
            "- All numbers must be floats with 2 decimal places (1.00 not 1)\n"
            "- Use exactly 2 spaces between numbers\n"
            "- Image coordinates: origin at top-left, X right, Y down\n\n"
            "- The appearance description should be a single WORD token that captures the key visual elements and style of the object. If no appearance change is evident from the user_edit, it must return 'null'.\n\n"
            "Example output:\n"
            "MATRIX\n"
            "[[0.88 0.00 0.00]\n"
            " [0.00 0.88 0.00]\n"
            " [0.00 0.00 1.00]]\n"
            "OBJECT_ID: 1\n"
            "APPEARANCE: <TOKEN>\n"
            "If user does not want to change the appearance of the object, APPEARANCE: null\n"
            "This is important, do not change the appearance of the object if the user does not want to change it. APPEARANCE field can be the name of a color, texture, or any other visual property."
            "If user does not want to change the appearance of the object appearance field should be null like this APPEARANCE: null\n",
        },
        {"role": "user", "content": user_edit},
    ]

    if model_name == "deepseek_r1_distill_qwen_32B":
        return messages_deepseek_r1_distill_qwen_32B_ONUR
    elif model_name == "qwen2_5_math_7b_instruct":
        return messages_qwen2_5_math_7b_instruct
    else:
        raise ValueError(f"Model {model_name} not supported")


def run_math_llm(model_name, model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device, logger):

    # 1. generate prompt
    messages = generate_prompt(model_name, user_edit, objects, scene_desc, spatial_rel, device)
    # Try parsing up to 5 times
    for attempt in range(5):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        reasoning = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"Model Reasoning (Attempt {attempt + 1}):")

        try:
            if model_name == "deepseek_r1_distill_qwen_32B":
                matrix = parse_transformation_matrix_deepseek(reasoning)
                object_id = parse_object_id_deepseek(reasoning)
                # appearance_token = parse_appearance_token_deepseek(reasoning)
            elif model_name == "qwen2_5_math_7b_instruct":
                matrix = parse_transformation_matrix_qwen(reasoning)
                object_id = parse_object_id_qwen(reasoning)
                appearance_token = parse_appearance_token_qwen(reasoning)
            else:
                raise ValueError(f"Model {model_name} not supported")
            return matrix, object_id, appearance_token, reasoning
        except Exception as e:  # Catch specific exception for better error handling
            if attempt == 4:  # Last attempt
                logger.warning(f"Failed to parse matrix after 5 attempts: {str(e)}, using identity matrix and object_id=1")
                identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                return identity_matrix, 1, "null", reasoning
            logger.error(f"Failed to parse matrix on attempt {attempt + 1}: {str(e)}, retrying...")
            continue


def run_math_analysis(user_edit: str, file_path: str, model_name: str, model: Any, tokenizer: Any, device: str, logger):
    try:
        # 0. parse detection file with enhanced information
        file_dir = os.path.dirname(file_path)
        objects, scene_desc, spatial_rel = parse_detection_file(file_path)

        # 2. run math llm
        matrix_array, object_id, appearance_token, reasoning = run_math_llm(
            model_name, model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device, logger
        )

        logger.info("Parsed Matrix: \n" + str(matrix_array))
        logger.info("Object ID:" + str(object_id))
        logger.info("Appearance Token:" + str(appearance_token))
        # 3. save files
        TRANSFORMATION_MATRIX_FILE = f"{file_dir}/transformation_matrix.npy"
        np.save(TRANSFORMATION_MATRIX_FILE, matrix_array)

        REASONING_FILE = f"{file_dir}/math_reasoning.txt"
        with open(REASONING_FILE, "w") as f:
            f.write(reasoning)

        OBJECT_ID_FILE = f"{file_dir}/object_id.txt"
        with open(OBJECT_ID_FILE, "w") as f:
            f.write(str(object_id))

        # 4. save appearance token
        ENHANCED_FILE_DESCRIPTION = f"{file_dir}/analysis_enhanced.txt"
        with open(ENHANCED_FILE_DESCRIPTION, "r") as token_file:
            analysis_lines = token_file.readlines()

        for i, line in enumerate(analysis_lines):
            if f"Object {object_id}:" in line:
                appearance_token_line = f"  Appearance Token: {appearance_token}\n"
                analysis_lines.insert(i + 2, appearance_token_line)
                break

        with open(ENHANCED_FILE_DESCRIPTION, "w") as token_file:
            token_file.writelines(analysis_lines)

        APPEARANCE_TOKEN_FILE = f"{file_dir}/appearance_token.txt"
        with open(APPEARANCE_TOKEN_FILE, "w") as f:
            f.write(str(appearance_token))

        return matrix_array, object_id, appearance_token

    except Exception as e:
        logging.error(f"Error in mathematical analysis for {file_path}: {str(e)}")
        raise
