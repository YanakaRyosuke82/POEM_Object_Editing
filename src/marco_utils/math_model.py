import torch
import logging
import os
import re
from typing import Dict, Any, List
import numpy as np



# Define transformation matrices with clear mathematical formulas
class TransformationMatrices:
    @staticmethod
    def translation(tx, ty):
        return np.array([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1]])
    
    @staticmethod
    def rotation(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    
    @staticmethod
    def scaling(sx, sy):
        return np.array([[sx, 0, 0],
                        [0, sy, 0],
                        [0, 0, 1]])
    
    @staticmethod
    def shear(shx, shy):
        return np.array([[1, shx, 0],
                        [shy, 1, 0],
                        [0, 0, 1]])



def parse_qwen_math_matrix(reasoning):
    """Extract and parse transformation matrix from model reasoning."""
    # Look for matrix pattern in the output format, including LaTeX style matrices
    # Find the output marker first
    output_start = reasoning.find("TRANSFORMATION_MATRIX")
    if output_start == -1:
        raise ValueError("Could not find output section in model reasoning")
    
    # Search for matrix patterns after the output marker
    reasoning_after_output = reasoning[output_start:]
    
    matrix_patterns = [
        # Standard array format
        r'\[\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]'
        r'\s*,\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]'
        r'\s*,\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*\]',
        # LaTeX style matrix pattern
        r'\\begin{pmatrix}\s*([-+]?\d*\.?\d+/?\d*)\s*&\s*([-+]?\d*\.?\d+/?\d*)\s*&\s*([-+]?\d*\.?\d+/?\d*)\s*\\\\'
        r'\s*([-+]?\d*\.?\d+/?\d*)\s*&\s*([-+]?\d*\.?\d+/?\d*)\s*&\s*([-+]?\d*\.?\d+/?\d*)\s*\\\\'
        r'\s*([-+]?\d*\.?\d+/?\d*)\s*&\s*([-+]?\d*\.?\d+/?\d*)\s*&\s*([-+]?\d*\.?\d+/?\d*)\s*\\end{pmatrix}'
    ]

    matrix_values = None
    for pattern in matrix_patterns:
        match = re.search(pattern, reasoning_after_output)
        if match:
            matrix_values = match.groups()
            break
    
    if not matrix_values:
        raise ValueError("Could not find transformation matrix in model output section")
    
    # Convert fractions (like sqrt(2)/2) to decimal numbers
    def convert_fraction(s):
        if '/' in s:
            if 'sqrt(2)' in s or '\\sqrt{2}' in s:
                return np.sqrt(2)/2
            num, denom = s.split('/')
            return float(num) / float(denom)
        return float(s)
    
    # Convert all values to floats
    matrix = np.array([convert_fraction(val) for val in matrix_values]).reshape(3, 3)
    
    return matrix



def parse_transformation_matrix(reasoning):
    """Extract and parse transformation matrix from model reasoning."""
    # Find the start and end tokens
    start_token = "<<MATRIX_START>>"
    end_token = "<<MATRIX_END>>"
    
    start_idx = reasoning.find(start_token)
    end_idx = reasoning.find(end_token, start_idx)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find the transformation matrix in the reasoning output.")
    
    # Extract the text between the tokens
    matrix_text = reasoning[start_idx + len(start_token):end_idx].strip()
    
    # Use regex to extract numbers, handling both integer and float formats
    pattern = r'[-+]?(?:\d*\.\d+|\d+\.?)'
    numbers = re.findall(pattern, matrix_text)
    
    if len(numbers) < 9:
        raise ValueError(f"Could not find 9 numbers in the matrix text. Found: {len(numbers)}")
        
    try:
        # Convert strings to floats and reshape into 3x3 matrix
        # Ensure float format by adding .0 to integers
        matrix_values = []
        for n in numbers[:9]:
            if '.' not in n:
                n = n + '.0'
            matrix_values.append(float(n))
            
        matrix = np.array(matrix_values, dtype=float).reshape(3, 3)
    except Exception as e:
        raise ValueError(f"Error parsing the matrix: {e}")
    
    # Validate the matrix shape
    if matrix.shape != (3, 3):
        raise ValueError("The parsed matrix is not a 3x3 matrix.")
    
    return matrix

def parse_object_id(reasoning):
    """Extract and parse object ID (integer) from model reasoning."""
    # Find the start and end tokens
    start_token = "<<OBJECT_ID_START>>"
    end_token = "<<OBJECT_ID_END>>"
    
    start_idx = reasoning.find(start_token)
    end_idx = reasoning.find(end_token, start_idx)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find the object ID in the reasoning output.")
    
    # Extract text between tokens and parse as integer
    id_text = reasoning[start_idx + len(start_token):end_idx].strip()
    try:
        # Remove "Object #" prefix if present and convert to int
        id_text = id_text.replace("Object #", "").strip()
        object_id = int(id_text)
        return object_id
    except ValueError:
        raise ValueError(f"Could not parse object ID from text: {id_text}")

def get_transformation_matrix(model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device):
    """Get transformation matrix from model reasoning."""
    # Create context from scene information
    scene_context = f"""Scene Information:
    {scene_desc}
    
    Spatial Relationships:
    {spatial_rel}
    
    Object Dimensions:"""
    
    for obj in objects:
        scene_context += f"""
        Object ID       : {obj['id']}
        Object class    : {obj['class']}
        Width          : {obj['width']:.3f}
        Height         : {obj['height']:.3f}
        Center         : ({obj['center'][0]:.3f}, {obj['center'][1]:.3f})
        ----------------------"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a math expert. Your task is to output a 3x3 transformation matrix and the object ID.\n\n"
                "IMPORTANT: Your response MUST follow this EXACT format:\n"
                "1. First explain your reasoning in max 50 words.\n"
                "2. Then output EXACTLY ONE matrix between these tokens:\n"
                "<<MATRIX_START>>\n"
                "[[x.xx  x.xx  x.xx]\n"
                " [x.xx  x.xx  x.xx]\n"
                " [x.xx  x.xx  x.xx]]\n"
                "<<MATRIX_END>>\n\n"
                "RULES:\n"
                "- Matrix must be 3x3\n" 
                "- All numbers must be floats (e.g. 1.0 not 1)\n"
                "- Use exactly 2 spaces between numbers\n"
                "- Image origin is top-left corner\n"
                "- X axis goes right, Y axis goes down\n\n"
                "Available transformations:\n"
                "1. Translation: [[1 0 tx], [0 1 ty], [0 0 1]]\n"
                "2. Rotation: [[cos(θ) -sin(θ) 0], [sin(θ) cos(θ) 0], [0 0 1]]\n"
                "3. Scale: [[sx 0 0], [0 sy 0], [0 0 1]]\n"
                "4. Shear: [[1 shx 0], [shy 1 0], [0 0 1]]\n"
                "5. Flip: [[-1 0 0], [0 1 0], [0 0 1]]\n\n"
                "Scene information:\n"
                "{scene_context}"
                "The transformation matrix must be a 3x3 numpy array with float values, you must use the tokens <<MATRIX_START>> and <<MATRIX_END>> to indicate the start and end of the matrix, do not make any mistake in the format."
                "Remember: you must use the tokens <<MATRIX_START>> and <<MATRIX_END>> to indicate the start and end of the matrix, do not make any mistake in the format."
                "Don't forget to use the tokens <<MATRIX_START>> and <<MATRIX_END>> to indicate the start and end of the matrix, do not make any mistake in the format."
                "The object ID is the one that is being subject to transformation. Write it as an integer between <<OBJECT_ID_START>> and <<OBJECT_ID_END>> tokens."
                "For example: <<OBJECT_ID_START>>1<<OBJECT_ID_END>> for object #1."
            )
        },
        {"role": "user", "content": user_edit}
    ]

    # Try parsing up to 5 times
    for attempt in range(5):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        reasoning = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.info(f"Model Reasoning (Attempt {attempt + 1}):")
        logging.info(reasoning)
        
        try:
            matrix, reasoning = parse_transformation_matrix(reasoning), reasoning
            object_id = parse_object_id(reasoning)
            return matrix, object_id, reasoning
        except:
            if attempt == 4:  # Last attempt
                logging.error("Failed to parse matrix after 5 attempts")
                raise
            logging.warning(f"Failed to parse matrix on attempt {attempt + 1}, retrying...")
            continue


def parse_detection_file(file_path):
    """Parse detection file and extract object information."""
    with open(file_path, 'r') as f:
        detection_data = f.read()

    # Extract scene description and spatial relationships
    scene_desc_match = re.search(r'Scene Description:\n(.*?)\n', detection_data)
    spatial_rel_match = re.search(r'Spatial Relationships:\n(.*?)\n', detection_data)
    
    scene_desc = scene_desc_match.group(1) if scene_desc_match else ""
    spatial_rel = spatial_rel_match.group(1) if spatial_rel_match else ""

    objects = []
    pattern = r'Object (\d+):\n\s+Class: (.*?)\n\s+Bounding Box.*?xmin=([\d.]+), ymin=([\d.]+), xmax=([\d.]+), ymax=([\d.]+)'
    
    for match in re.finditer(pattern, detection_data):
        xmin, ymin, xmax, ymax = map(float, match.groups()[2:])
        
        corners = np.array([
            [xmin, ymin, 1],  # top-left
            [xmax, ymin, 1],  # top-right
            [xmax, ymax, 1],  # bottom-right
            [xmin, ymax, 1]   # bottom-left
        ])
        
        obj = {
            'id': int(match.group(1)),
            'class': match.group(2),
            'bbox': [xmin, ymin, xmax, ymax],
            'corners': corners,
            'width': xmax - xmin,
            'height': ymax - ymin,
            'center': [(xmax + xmin)/2, (ymax + ymin)/2]
        }
        objects.append(obj)
    
    return objects, scene_desc, spatial_rel



def run_math_analysis(user_edit: str, file_path: str, img_path: str, model: Any, 
                     tokenizer: Any, device: str):
    """
    Run mathematical analysis on a single sample
    
    Args:
        file_path: Path to analysis_enhanced.txt file
        img_path: Path to original image
        model: Qwen model instance
        tokenizer: Qwen tokenizer instance
        device: Computing device
    """
    try:
        # Parse detection file with enhanced information
        file_dir = os.path.dirname(file_path)
        objects, scene_desc, spatial_rel = parse_detection_file(file_path)

        
    
        # Get transformation matrix with scene context
        matrix_array, object_id, reasoning = get_transformation_matrix(model, tokenizer, user_edit, objects, scene_desc, spatial_rel, device)
        
        logging.info("Parsed Matrix:")
        logging.info(matrix_array)
        
        # Store transformation matrix to file
        TRANSFORMATION_MATRIX_FILE = f'{file_dir}/transformation_matrix.npy'
        np.save(TRANSFORMATION_MATRIX_FILE, matrix_array)
        logging.info(f"Transformation matrix saved to {TRANSFORMATION_MATRIX_FILE}")
        
        # Store reasoning to file
        REASONING_FILE = f'{file_dir}/math_reasoning.txt'
        with open(REASONING_FILE, 'w') as f:
            f.write(reasoning)
        logging.info(f"Reasoning saved to {REASONING_FILE}")
        
        # Store object ID to file
        OBJECT_ID_FILE = f'{file_dir}/object_id.txt'
        with open(OBJECT_ID_FILE, 'w') as f:
            f.write(str(object_id))
        logging.info(f"Object ID saved to {OBJECT_ID_FILE}")
        
        return matrix_array, object_id
        
    except Exception as e:
        logging.error(f"Error in mathematical analysis for {file_path}: {str(e)}")
        raise 