import torch
import numpy as np
import cv2
import os
import re
import logging
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from ultralytics import SAM
def parse_detection_file(file_path: str, img_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict]]:
    """
    Parse detection file to extract points and bounding boxes organized by class and object ID
    
    Args:
        file_path: Path to analysis.txt file
        img_path: Path to corresponding image
        
    Returns:
        Tuple of (input_points_by_class, input_labels_by_class, input_boxes_by_class, objects)
        where each *_by_class is a dict mapping class names to numpy arrays
    """
    # Get image dimensions
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    with open(file_path, 'r') as f:
        content = f.read()

    objects = []
    object_blocks = re.findall(r'Object (\d+):(.*?)(?=Object \d+:|$)', content, re.DOTALL)

    # Track points, labels and boxes by class
    points_by_class = {}
    labels_by_class = {}
    boxes_by_class = {}

    for obj_id, obj_block in object_blocks:
        class_match = re.search(r'Class:\s*(\w+)', obj_block)
        if not class_match:
            continue
            
        class_name = class_match.group(1).strip()
        bbox_match = re.search(r'Bounding Box \(normalized\):\s*xmin=([\d.]+),\s*ymin=([\d.]+),\s*xmax=([\d.]+),\s*ymax=([\d.]+)', obj_block)
        if not bbox_match:
            continue

        xmin, ymin, xmax, ymax = map(float, bbox_match.groups())
        
        # Extract segmentation points if they exist
        seg_points = []
        point_matches = re.finditer(r'\(([\d.]+),\s*([\d.]+)\)', obj_block)
        for match in point_matches:
            x, y = map(float, match.groups())
            seg_points.append([x * img_width, y * img_height])

        # Calculate center point
        center_x = (xmin + xmax) / 2 * img_width
        center_y = (ymin + ymax) / 2 * img_height
        
        # Make bbox 30% bigger for better SAM segmentation results
        width = (xmax - xmin) * img_width
        height = (ymax - ymin) * img_height
        
        xmin = max(0, xmin * img_width - width * 0.15)
        ymin = max(0, ymin * img_height - height * 0.15)
        xmax = min(img_width, xmax * img_width + width * 0.15)
        ymax = min(img_height, ymax * img_height + height * 0.15)
        
        obj_data = {
            'id': int(obj_id),
            'class': class_name,
            'center': [center_x, center_y],
            'bbox': [xmin, ymin, xmax, ymax],
            'seg_points': seg_points
        }
        objects.append(obj_data)

        # Initialize arrays for new classes
        class_key = f"{class_name}_{obj_id}"
        if class_key not in points_by_class:
            points_by_class[class_key] = []
            labels_by_class[class_key] = []
            boxes_by_class[class_key] = []

        # Add center point and box
        points_by_class[class_key].append([center_x, center_y])
        labels_by_class[class_key].append(1)  # 1 indicates foreground point
        boxes_by_class[class_key].append([xmin, ymin, xmax, ymax])

        # Add segmentation points
        if seg_points:
            points_by_class[class_key].extend(seg_points)
            labels_by_class[class_key].extend([1] * len(seg_points))

    # Convert lists to numpy arrays
    for class_key in points_by_class:
        points_by_class[class_key] = np.array(points_by_class[class_key])
        labels_by_class[class_key] = np.array(labels_by_class[class_key])
        boxes_by_class[class_key] = np.array(boxes_by_class[class_key])

    return points_by_class, labels_by_class, boxes_by_class, objects

def run_sam_refine(file_analysis_path: str, img_path: str, sam_model: SAM):
    """
    Run SAM refinement on detected objects
    
    Args:
        file_analysis_path: Path to analysis file
        img_path: Path to image
        sam_model: Loaded SAM model
    """
    output_dir = os.path.dirname(file_analysis_path)
    os.makedirs(output_dir, exist_ok=True)

    # Parse detection file
    input_points_by_class, input_labels_by_class, input_boxes_by_class, objects = parse_detection_file(file_analysis_path, img_path)

    # Get original content for preserving text sections
    with open(file_analysis_path, 'r') as f:
        content = f.read()

    SAM_MASKS = {}
    # Run SAM inference for each object
    sam_bboxes = {}
    for class_key in input_points_by_class:
        points = input_points_by_class[class_key]
        labels = input_labels_by_class[class_key]
        boxes = input_boxes_by_class[class_key]

        # Run SAM for this object
        results = sam_model(img_path, points=points.tolist(), labels=labels.tolist())

        if len(results) == 0 or len(results[0].masks.data) == 0:
            logging.warning(f"No mask found for {class_key}")
            continue

        # Get mask for this object
        mask_array = results[0].masks.data[0].cpu().numpy()

        # Get bounding box from mask
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)

        if not np.any(rows) or not np.any(cols):
            logging.warning(f"Empty mask for {class_key}")
            continue

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Save visualization and mask
        class_name = class_key.rsplit('_', 1)[0]
        obj_id = class_key.rsplit('_', 1)[1]
        plt.imsave(os.path.join(output_dir, f'mask_{obj_id}.png'), mask_array, cmap='gray')
        SAM_MASKS[obj_id] = mask_array
        # Store normalized coordinates
        mask_bbox = {
            'xmin': float(xmin) / mask_array.shape[1],
            'ymin': float(ymin) / mask_array.shape[0],
            'xmax': float(xmax) / mask_array.shape[1],
            'ymax': float(ymax) / mask_array.shape[0]
        }
        sam_bboxes[class_key] = mask_bbox

    # Write enhanced analysis file
    enhanced_file = os.path.join(output_dir, "analysis_enhanced.txt")
    with open(enhanced_file, 'w') as f:
        f.write("=== ENHANCED DETECTION RESULTS ===\n\n")
        f.write("Detected Objects:\n")

        for i, obj in enumerate(objects):
            class_key = f"{obj['class']}_{obj['id']}"
            f.write(f"Object {i+1}:\n")
            f.write(f"  Class: {obj['class']}\n")

            if class_key in sam_bboxes:
                bbox = sam_bboxes[class_key]
                f.write(f"  Bounding Box (normalized): xmin={bbox['xmin']:.3f}, ymin={bbox['ymin']:.3f}, "
                       f"xmax={bbox['xmax']:.3f}, ymax={bbox['ymax']:.3f}\n")
                width = bbox['xmax'] - bbox['xmin']
                height = bbox['ymax'] - bbox['ymin']
                f.write(f"  Bounding Box (SLD format): [{bbox['xmin']:.3f}, {bbox['ymin']:.3f}, "
                       f"{width:.3f}, {height:.3f}]\n")

            if obj['seg_points']:
                f.write("  Segmentation Points:\n")
                for point in obj['seg_points']:
                    f.write(f"    ({point[0]/mask_array.shape[1]:.3f}, {point[1]/mask_array.shape[0]:.3f})\n")

        # Preserve original text sections
        for section in ['Scene Description', 'Spatial Relationships', 'Background Description', 'Generation Prompt']:
            section_match = re.search(f'{section}:\n(.*?)(?=\n\n|\n[A-Z]|$)', content, re.DOTALL)
            if section_match:
                f.write(f"\n{section}:\n{section_match.group(1).strip()}\n")
    return SAM_MASKS    
