import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Tuple, Any
import logging
import cv2
import matplotlib.pyplot as plt
import os

def parse_line(line: str, objects: list) -> None:
    """
    Parse a single line of model output with error handling
    
    Args:
        line: Single line from model output
        objects: List to store parsed objects
    """
    try:

        line = line.strip()
        print(line)
        if not line:
            return None, None
            
        if line.startswith('DETECT:'):
            _, detection = line.split(':', 1)
            parts = detection.strip().split('|')
            if len(parts) != 6:
                logging.warning(f"Skipping malformed DETECT line: {line}")
                return
            
            try:
                obj_id = int(parts[0])
                xmin, ymin, xmax, ymax = map(float, parts[2:])
                
                while len(objects) < obj_id:
                    objects.append({'class': None, 'bbox': None, 'point': None})
                objects[obj_id-1].update({
                    'class': parts[1],
                    'bbox': [xmin, ymin, xmax, ymax]
                })
            except (ValueError, IndexError) as e:

                logging.warning(f"Error parsing DETECT values: {e}")
                
        elif line.startswith('POINTS:'):
            _, points_data = line.split(':', 1)
            parts = points_data.strip().split('|')
            if len(parts) != 3:
                logging.warning(f"Skipping malformed POINTS line: {line}")
                return
                
            try:
                obj_id = int(parts[0])
                x, y = map(float, parts[1:])
                if 0 <= obj_id-1 < len(objects):
                    objects[obj_id-1]['point'] = (x, y)
            except (ValueError, IndexError) as e:
                logging.warning(f"Error parsing POINTS values: {e}")
                
    except Exception as e:
        logging.warning(f"Error parsing line '{line}': {e}")

def parse_image(image_path: str, model: Any, processor: Any, device: str, user_edit: str = None) -> Dict[str, Any]:
    """
    Parse an image using Qwen2VL model to detect objects, points, and scene understanding.
    
    Args:
        image_path: Path to input image
        model: Pre-loaded Qwen model instance
        processor: Pre-loaded processor instance
        device: Computing device ('cuda' or 'cpu')
        
    Returns:
        Dict containing detected objects, scene description, and spatial relationships
    """
    # Prepare prompt
    messages = [
        {"role": "system", "content": """You are a leading computer vision expert specializing in object detection, scene understanding, and spatial relationships.
For each detected object in the image, output exactly one line in the following format:
DETECT: <object_id_number>|<object_class>|<xmin>|<ymin>|<xmax>|<ymax>
POINTS: <object_id_number>|<x_center>|<y_center>
SCENE: <scene_description>
SPATIAL: <spatial_relationships>
BACKGROUND: <background_description>
GENERATION: <generation_prompt>
Note: provide:
- Bounding boxes use normalized coordinates: [0, 0] at top-left and [1, 1] at bottom-right.
- Points share the same coordinate system; provide x_center and y_center.
- Use consistent numerical object IDs for detections and points.
- The generation prompt should be a concise prompt for image editing that captures the key visual elements and style, incorporating the user's edit request: {user_edit}
        """},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Analyze this image, detect objects, provide keypoints, describe scene and spatial relationships."}
        ]}
    ]

    # Process image
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Generate output
    with torch.cuda.device(device):
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                        padding=True, return_tensors="pt").to(device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    try:
        print(f"VLM OUTPUT TEXT: {output_text}")
        # Parse output
        objects = []
        scene_desc = "a realistic image"  # Default fallback
        spatial_rel = "a realistic image"  # Default fallback
        background_desc = "a realistic image"  # Default fallback
        generation_prompt = user_edit
        
        try:
            for line in output_text.split('\n'):
                try:
                    if line.startswith('DETECT:') or line.startswith('POINTS:'):
                        parse_line(line, objects)
                    elif line.startswith('SCENE:'):
                        _, scene_desc = line.split(':', 1)
                        if not scene_desc.strip():
                            scene_desc = "a realistic image"
                    elif line.startswith('SPATIAL:'):
                        _, spatial_rel = line.split(':', 1)
                        if not spatial_rel.strip():
                            spatial_rel = "a realistic image"
                    elif line.startswith('BACKGROUND:'):
                        _, background_desc = line.split(':', 1)
                        if not background_desc.strip():
                            background_desc = "a realistic image"
                except ValueError:
                    continue  # Skip malformed lines
                except Exception as e:
                    print(f"Error parsing line: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error processing output text: {str(e)}")

        # Filter out any None or incomplete objects
        objects = [obj for obj in objects if obj['class'] is not None and obj['bbox'] is not None]

        try:
            # Debug: Visualize detected objects on the image
            debug_image = cv2.imread(image_path)
            if debug_image is None:
                raise ValueError("Failed to load image for debugging")
                
            height, width = debug_image.shape[:2]
            for obj in objects:
                try:
                    if obj['bbox']:
                        xmin, ymin, xmax, ymax = [int(coord * width) if i % 2 == 0 else int(coord * height) 
                                                for i, coord in enumerate(obj['bbox'])]
                        cv2.rectangle(debug_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(debug_image, obj['class'], (xmin, ymin-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if obj['point']:
                        x, y = obj['point']
                        px = int(x * width)
                        py = int(y * height)
                        cv2.circle(debug_image, (px, py), 5, (255, 0, 0), -1)
                except Exception as e:
                    print(f"Error drawing object: {str(e)}")
                    continue

            plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            plt.title("Debug: Detected Objects")
            plt.axis('off')
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            debug_image = None

        return {
            'objects': objects,
            'scene_description': scene_desc.strip(),
            'spatial_relationships': spatial_rel.strip(),
            'background_description': background_desc.strip(),
            'generation_prompt': generation_prompt.strip(),
            'debug_image': debug_image
        }

    except Exception as e:
        print(f"Critical error in parse_image: {str(e)}")
        return {
            'objects': [],
            'scene_description': "a realistic image",
            'spatial_relationships': "a realistic image", 
            'background_description': "a realistic image",
            'generation_prompt': user_edit if user_edit else "a realistic image",
            'debug_image': None
        }






def save_results_image_parse(sample_dir: str, image: cv2.Mat, original_path: str, results: dict):
    """
    Save processed image and analysis results in the exact specified format
    
    Args:
        sample_dir: Directory to save results
        image: Processed image
        original_path: Path to original image
        results: Dictionary containing analysis results
    """
    try:
        # Save images
        cv2.imwrite(os.path.join(sample_dir, "edited.png"), image)
        cv2.imwrite(os.path.join(sample_dir, "original.png"), cv2.imread(original_path))
        
        # Save analysis with exact formatting
        with open(os.path.join(sample_dir, "analysis.txt"), 'w') as f:
            f.write("=== DETECTION RESULTS ===\n\n")
            f.write("Detected Objects:\n")
            
            for i, obj in enumerate(results['objects']):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {obj['class']}\n")
                bbox = obj['bbox']
                f.write(f"  Bounding Box (normalized): xmin={bbox[0]:.3f}, ymin={bbox[1]:.3f}, "
                       f"xmax={bbox[2]:.3f}, ymax={bbox[3]:.3f}\n")
                if obj['point']:
                    f.write(f"  Segmentation Point: ({obj['point'][0]:.3f}, {obj['point'][1]:.3f})\n")
                f.write("\n")
            
            f.write("Scene Description:\n")
            f.write(f"{results['scene_description']}\n\n")
            
            f.write("Spatial Relationships:\n")
            f.write(f"{results['spatial_relationships']}\n\n")

            f.write("Background Description:\n")
            f.write(f"{results['background_description']}\n\n")

            f.write("Generation Prompt:\n")
            f.write(f"{results['generation_prompt']}\n\n")
            
            f.write("=====================\n")

    
        
    
  
            
    except Exception as e:
        logging.error(f"Error saving results to {sample_dir}: {str(e)}")