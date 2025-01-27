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



def parse_matrix_from_reasoning(reasoning):
    """Extract and parse transformation matrix from model reasoning."""
    # Look for matrix pattern in the output format, including LaTeX style matrices
    # Find the output marker first
    output_start = reasoning.find("```output")
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

def get_transformation_matrix(model, tokenizer, user_edit, objects, scene_desc, spatial_rel):
    """Get transformation matrix from model reasoning."""
    # Create context from scene information
    scene_context = f"""Scene Information:
    {scene_desc}
    
    Spatial Relationships:
    {spatial_rel}
    
    Object Dimensions:"""
    
    for obj in objects:
        scene_context += f"""
    {obj['class']}:
        Width: {obj['width']:.3f}
        Height: {obj['height']:.3f}
        Center: ({obj['center'][0]:.3f}, {obj['center'][1]:.3f})"""

    messages = [
        {"role": "system", "content": "Integrate natural language reasoning with programs to solve user query. Given the context information and object dimensions below, determine the appropriate transformation matrix for the requested edit.\n\n"
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
                                    "   Example: Translation + Rotation = [[cos(θ) -sin(θ) tx][sin(θ) cos(θ) ty][0 0 1]]\n\n"   
        },
        {"role": "user", "content": user_edit}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    reasoning = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    logging.info("Model Reasoning:")
    # logging.info(reasoning)
    
    return parse_matrix_from_reasoning(reasoning), reasoning


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
        matrix_array, reasoning = get_transformation_matrix(model, tokenizer, user_edit, objects, scene_desc, spatial_rel)
        
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
        
        return matrix_array
        
    except Exception as e:
        logging.error(f"Error in mathematical analysis for {file_path}: {str(e)}")
        raise 