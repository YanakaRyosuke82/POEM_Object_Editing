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


def parse_object_id_qwen(reasoning):
    """Extract and parse object ID (integer) from model reasoning."""
    # Try different patterns for object ID
    patterns = [
        # Standard format
        r"OBJECT_ID:\s*(\d+)",
        # LaTeX boxed format
        r"\\boxed{.*?,\s*(\d+)}",
        # Plain number after comma in LaTeX
        r"\\end{pmatrix},\s*(\d+)",
        # Number after comma
        r"matrix\},\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, reasoning)
        if match:
            try:
                object_id = int(match.group(1))
                return object_id
            except ValueError:
                continue

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

    messages_deepseek_r1_distill_qwen_32B = [
        {
            "role": "system",
            "content": (
                "You are a computer vision math expert. Your task is to output a 3x3 transformation matrix and object ID based on the user's edit request.\n\n"
                "OUTPUT FORMAT REQUIREMENTS:\n"
                "1. Brief reasoning (max 2 lines)\n"
                "2. Object ID in format: <<OBJECT_ID_START>>N<<OBJECT_ID_END>> where N is the integer ID\n"
                "3. Matrix in format:\n"
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
                "AVAILABLE TRANSFORMATIONS:\n"
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

    mess1 = [
        {
            "role": "system",
            "content": "Integrate natural language reasoning with programs to solve user query. Given the scene content and the user edit, determine the appropriate transformation matrix for the requested edit; and the object ID.\n\n"
            "Scene content: " + scene_context + "\n\n"
            "List of possible operations:\n"
            "1. Translation: Moving objects in x,y directions\n"
            "   Example: [[1 0 tx][0 1 ty][0 0 1]]\n\n"
            "2. Rotation: Rotating objects by angle θ\n"
            "   Example: [[cos(θ) -sin(θ) 0][sin(θ) cos(θ) 0][0 0 1]]\n\n"
            "3. Scaling: Changing object size\n"
            "   Example: [[sx 0 0][0 sy 0][0 0 1]]\n\n"
            "4. Shear: Skewing objects\n"
            "   Example: [[1 shx 0][shy 1 0][0 0 1]]\n\n"
            "5. Combined transformations are also allowed:\n"
            "   Example: multiply the transformation matrices corresponding to the operations. for example translation + rotation = [[cos(θ) -sin(θ) tx][sin(θ) cos(θ) ty][0 0 1]] * [[1 0 tx][0 1 ty][0 0 1]]; additional examples: translation + scaling = [[1 0 tx][0 1 ty][0 0 1]] * [[sx 0 0][0 sy 0][0 0 1]], translation + rotation + scaling = [[cos(θ) -sin(θ) tx][sin(θ) cos(θ) ty][0 0 1]] * [[sx 0 0][0 sy 0][0 0 1]]   \n\n"
            " NOTE: I need the output matrix as a numpy array; and the output object ID as an integer\n"
            "[[0.88 0.  0. ]\n"
            " [0.  0.88 0. ]\n"
            " [0.  0.  1. ]]  \n\n",
        },
        {"role": "user", "content": user_edit},
    ]

    messages_qwen2_5_math_7b_instruct = [
        {
            "role": "system",
            "content": (
                "Integrate natural language reasoning with programs to solve the problem above. You are a mathematical reasoning assistant that generates transformation matrices for image editing operations. Your task is to analyze the edit request and output a precise 3x3 transformation matrix and object ID.\n\n"
                "Scene Context with Object IDs:\n" + scene_context + "\n\n"
                "Output Format Requirements:\n"
                "1. Provide clear mathematical reasoning explaining your approach\n"
                "2. Use these exact output markers:\n\n"
                "OBJECT_ID: <number>\n"
                "TRANSFORMATION_MATRIX:\n"
                "[[a.aa  b.bb  c.cc]\n"
                " [d.dd  e.ee  f.ff]\n"
                " [g.gg  h.hh  i.ii]]\n\n"
                "Matrix Requirements:\n"
                "- Must be 3x3 homogeneous transformation matrix\n"
                "- Use exactly 2 spaces between numbers\n"
                "- All numbers must be floats with 2 decimal places (e.g. 1.00)\n"
                "- Follow numpy array format\n"
                "- Image coordinates: origin at top-left, +X right, +Y down\n\n"
                "Available Transformation Templates:\n"
                "1. Translation (move by tx, ty):\n"
                "   [[1.00  0.00  tx]\n"
                "    [0.00  1.00  ty]\n"
                "    [0.00  0.00  1.00]]\n\n"
                "2. Rotation (by angle θ):\n"
                "   [[cos(θ)  -sin(θ)  0.00]\n"
                "    [sin(θ)   cos(θ)  0.00]\n"
                "    [0.00     0.00    1.00]]\n\n"
                "3. Scale (by factors sx, sy):\n"
                "   [[sx    0.00  0.00]\n"
                "    [0.00  sy    0.00]\n"
                "    [0.00  0.00  1.00]]\n\n"
                "4. Shear (by factors shx, shy):\n"
                "   [[1.00  shx   0.00]\n"
                "    [shy   1.00  0.00]\n"
                "    [0.00  0.00  1.00]]\n\n"
                "5. Flip X:\n"
                "   [[-1.00  0.00  0.00]\n"
                "    [0.00   1.00  0.00]\n"
                "    [0.00   0.00  1.00]]\n\n"
                "SCENE CONTEXT:\n"
                f"{scene_context}\n\n"
                "IMPORTANT:\n"
                "- Object ID must be from the provided OBJECT DETAILS list\n"
                "- Matrices can be combined by multiplication for compound transformations\n"
                "- Ensure mathematical correctness and precise decimal formatting\n"
            ),
        },
        {"role": "user", "content": user_edit},
    ]
    if model_name == "deepseek_r1_distill_qwen_32B":
        return messages_deepseek_r1_distill_qwen_32B
    elif model_name == "qwen2_5_math_7b_instruct":
        return messages_qwen2_5_math_7b_instruct
    else:
        raise ValueError(f"Model {model_name} not supported")


def run_math_llm(model_name, model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device):

    # 1. generate prompt
    messages = generate_prompt(model_name, user_edit, objects, scene_desc, spatial_rel, device)
    # Try parsing up to 5 times
    for attempt in range(5):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        reasoning = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.info(f"Model Reasoning (Attempt {attempt + 1}):")

        try:
            if model_name == "deepseek_r1_distill_qwen_32B":
                matrix = parse_transformation_matrix_deepseek(reasoning)
                object_id = parse_object_id_deepseek(reasoning)
            elif model_name == "qwen2_5_math_7b_instruct":
                matrix = parse_transformation_matrix_qwen(reasoning)
                object_id = parse_object_id_qwen(reasoning)
            else:
                raise ValueError(f"Model {model_name} not supported")
            return matrix, object_id, reasoning
        except:
            if attempt == 4:  # Last attempt
                logging.error("Failed to parse matrix after 5 attempts")
                raise
            logging.warning(f"Failed to parse matrix on attempt {attempt + 1}, retrying...")
            continue


def run_math_analysis(user_edit: str, file_path: str, model_name: str, model: Any, tokenizer: Any, device: str):
    try:
        # 0. parse detection file with enhanced information
        file_dir = os.path.dirname(file_path)
        objects, scene_desc, spatial_rel = parse_detection_file(file_path)

        # 2. run math llm
        matrix_array, object_id, reasoning = run_math_llm(model_name, model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device)

        logging.info("Parsed Matrix: \n" + str(matrix_array))
        logging.info("Parsed Matrix: \n" + str(matrix_array))
        logging.info("Object ID:" + str(object_id))

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
