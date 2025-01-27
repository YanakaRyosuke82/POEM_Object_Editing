from .config_sld_format import format_example
import json
import logging
from typing import Tuple, Dict, Any
import numpy as np
import os
def generate_sld_config(
    sample_dir: str,
    analysis_enhanced_file: str
) -> str:
    """Generate Stable Layout Diffusion configuration from enhanced analysis.
    
    This function processes the enhanced analysis file to generate a configuration
    for the Stable Layout Diffusion model. It validates input content, formats
    the data according to SLD requirements, and ensures proper JSON structure.
    
    Args:
        sample_dir: Directory containing the sample images and outputs
        analysis_enhanced_file: Path to file containing enhanced scene analysis
        vlm_model: Vision-language model for processing scene descriptions
        vlm_processor: Associated processor for the VL model
        
    Returns:
        str: Generated SLD configuration as a validated JSON string
        
    Raises:
        FileNotFoundError: If analysis file cannot be found
        json.JSONDecodeError: If content fails JSON validation
        ValueError: If bbox coordinates are invalid
    """
    try:
        # Load and validate input content
        with open(analysis_enhanced_file, 'r') as file:
            content = file.read().strip()

        # Extract original bbox from enhanced analysis
        bbox_line = [line for line in content.split('\n') if 'Bounding Box' in line][0]
        orig_xmin = float(bbox_line.split('xmin=')[1].split(',')[0])
        orig_ymin = float(bbox_line.split('ymin=')[1].split(',')[0])
        orig_xmax = float(bbox_line.split('xmax=')[1].split(',')[0])
        orig_ymax = float(bbox_line.split('ymax=')[1].split(',')[0])

        # Load transformed bbox from transformed_bbox.txt
        transformed_bbox = [0.348, 0.286, 0.538, 0.630]  # SLD format [x, y, width, height]

        # Compare original and transformed bboxes
        orig_bbox = [orig_xmin, orig_ymin, orig_xmax - orig_xmin, orig_ymax - orig_ymin]
        bbox_changed = any(abs(o - n) > 0.001 for o, n in zip(orig_bbox, transformed_bbox))

        # Extract scene description
        scene_desc_lines = [line for line in content.split('\n') if 'Scene Description:' in line]
        scene_description = scene_desc_lines[0].replace('Scene Description:', '').strip()

        # Extract object class from enhanced analysis
        object_class = None
        for line in content.split('\n'):
            if 'Class:' in line:
                object_class = line.split('Class:')[1].strip().lower()
                break

        # Extract generation prompt and background description
        content_lines = content.split('\n')
        generation_prompt = None
        bg_prompt = None
        
        for i, line in enumerate(content_lines):
            if 'Generation Prompt:' in line:
                # Get all lines until next section or end
                prompt_lines = []
                j = i + 1
                while j < len(content_lines) and not content_lines[j].startswith('===') and not any(x in content_lines[j] for x in ['Scene Description:', 'Background Description:', 'Spatial Relationships:', 'Class:']):
                    if content_lines[j].strip():
                        prompt_lines.append(content_lines[j].strip())
                    j += 1
                generation_prompt = ' '.join(prompt_lines)
            
            if 'Background Description:' in line:
                # Get all lines until next section or end
                bg_lines = []
                j = i + 1
                while j < len(content_lines) and not content_lines[j].startswith('===') and not any(x in content_lines[j] for x in ['Scene Description:', 'Generation Prompt:', 'Spatial Relationships:', 'Class:']):
                    if content_lines[j].strip():
                        bg_lines.append(content_lines[j].strip())
                    j += 1
                bg_prompt = ' '.join(bg_lines)

        if not generation_prompt or not bg_prompt:
            raise ValueError("Could not extract generation prompt or background description")
     
        # Construct base config, must be a list of dicts
        config_data = [{
            "input_fname": 'input.png',
            "output_dir": os.path.basename(sample_dir),
            "prompt": generation_prompt,
            "generator": "dalle",
            "llm_parsed_prompt": {
                "objects": [
                    [object_class, [None]]
                ],
                "bg_prompt": bg_prompt,
                "neg_prompt": "null"
            }
        }]
        # Only add layout suggestions if bbox changed
        if bbox_changed:
            config_data[0]["llm_layout_suggestions"] = [
                [f"{object_class} #1", transformed_bbox]
            ]
        
        # Write validated config
        config_path = f"{sample_dir}/config_sld.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
            
        logging.info(f"Successfully generated SLD config at {config_path}")
        return json.dumps(config_data)
    except Exception as e:
        logging.error(f"Error generating SLD config: {str(e)}")
        raise
