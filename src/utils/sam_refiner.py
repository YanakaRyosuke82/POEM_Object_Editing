import torch
import numpy as np
import cv2
import os
import re
import logging
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from utils.models import Models
import argparse
from ultralytics import SAM

def parse_detection_file(file_path: str, img_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse detection file to extract points and bounding boxes
    
    Args:
        file_path: Path to analysis.txt file
        img_path: Path to corresponding image
        
    Returns:
        Tuple of (input_points, input_labels, input_boxes)
    """

    # Parse bounding box coordinates from detection file
    with open(file_path, 'r') as f:
        content = f.read()

    # Get image dimensions
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # Extract bounding box coordinates and segmentation points for each object
    objects = []
    # Use regex to find all object blocks
    object_blocks = re.findall(r'Object \d+:(.*?)(?=Object \d+:|$)', content, re.DOTALL)

    for obj_block in object_blocks:
        # Extract class name
        class_match = re.search(r'Class:\s*(\w+)', obj_block)
        if class_match:
            class_name = class_match.group(1).strip()
            
            # Extract bounding box coordinates
            bbox_match = re.search(r'Bounding Box \(normalized\):\s*xmin=([\d.]+),\s*ymin=([\d.]+),\s*xmax=([\d.]+),\s*ymax=([\d.]+)', obj_block)
            if bbox_match:
                xmin, ymin, xmax, ymax = map(float, bbox_match.groups())
                
                # Extract segmentation points
                seg_points = []
                if 'Segmentation Point:' in obj_block:
                    point_match = re.search(r'\(([\d.]+),\s*([\d.]+)\)', obj_block)
                    if point_match:
                        x, y = map(float, point_match.groups())
                        seg_points.append([x * img_width, y * img_height])
                
                # Calculate center point and scale to image dimensions
                center_x = (xmin + xmax) / 2 * img_width
                center_y = (ymin + ymax) / 2 * img_height
                
                # Make bbox 30% bigger
                width = (xmax - xmin) * img_width
                height = (ymax - ymin) * img_height
                
                # Expand by 30% in each direction
                xmin = max(0, xmin * img_width - width * 0.15)
                ymin = max(0, ymin * img_height - height * 0.15)
                xmax = min(img_width, xmax * img_width + width * 0.15)
                ymax = min(img_height, ymax * img_height + height * 0.15)
                
                # Store center, bbox coordinates and segmentation points
                objects.append({
                    'class': class_name,
                    'center': [center_x, center_y],
                    'bbox': [xmin, ymin, xmax, ymax],
                    'seg_points': seg_points
                })

    # Convert to input format for SAM
    input_points = np.array([[obj['center'][0], obj['center'][1]] for obj in objects])
    input_labels = np.array([1] * len(objects))  # 1 indicates foreground point
    input_boxes = np.array([[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in objects])

    # Add segmentation points to input points and labels
    for obj in objects:
        if obj['seg_points']:
            seg_points = np.array(obj['seg_points'])
            input_points = np.vstack([input_points, seg_points])
            input_labels = np.append(input_labels, [1] * len(obj['seg_points']))

    print("Using points and bbox prompts together:")
    print("Points:", input_points)
    print("Labels:", input_labels)
    print("Boxes:", input_boxes)

    return input_points, input_labels, input_boxes, objects
def run_sam_refine(file_analysis_path: str, img_path: str, sam_model: Any):

    
    img_name = os.path.basename(img_path)
    output_dir = os.path.dirname(file_analysis_path)
    os.makedirs(output_dir, exist_ok=True)
    input_points, input_labels, input_boxes, objects = parse_detection_file(file_analysis_path, img_path)

    with open(file_analysis_path, 'r') as f:
        content = f.read()


    img_width, img_height = 512, 512

    # Load a model
    model = SAM("sam2.1_b.pt")

    # Display model information (optional)
    model.info()

    # Run inference with both points and bounding box prompts
    results = model(img_path, points=input_points.tolist(), labels=input_labels.tolist())

    # Store SAM bboxes for each object
    sam_bboxes = {}
    # Plot and save the results for each object
    for i, (r, obj) in enumerate(zip(results, objects)):
        print(f"Processing object {i+1}: {obj['class']}")
        
        # Get segmentation result and binary mask
        im_array = r.plot()  # Segmentation result
        im_array = im_array[:, :, ::-1]  # Convert BGR to RGB
        
        # Check if there are any masks
        if len(r.masks.data) == 0:
            print(f"Warning: No mask found for {obj['class']}")
            continue
            
        mask_array = r.masks.data[0].cpu().numpy()  # Binary mask
        
        # Get bounding box of binary mask
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print(f"Warning: Empty mask for {obj['class']}")
            continue
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Save visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(im_array)
        plt.title(f"{obj['class']} - Segmented")
        plt.axis('off')
        
        plt.subplot(132) 
        plt.imshow(mask_array, cmap='gray')
        plt.title(f"{obj['class']} - Mask")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(mask_array, cmap='gray')
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                           fill=False, color='red', linewidth=2))
        plt.title(f"{obj['class']} - Mask + Box")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sam_analysis_{i+1}.png'))
        plt.close()
        
        # Calculate normalized coordinates
        mask_bbox = {
            'xmin': float(xmin) / mask_array.shape[1],
            'ymin': float(ymin) / mask_array.shape[0], 
            'xmax': float(xmax) / mask_array.shape[1],
            'ymax': float(ymax) / mask_array.shape[0]
        }
        sam_bboxes[obj["class"]] = mask_bbox

        # Store binary mask separately
        mask_path = os.path.join(output_dir, f'mask_{i}.png')
        plt.imsave(mask_path, mask_array, cmap='gray')
        print(f"Binary mask saved at: {mask_path}")

    # Save results to single enhanced file
    enhanced_file = os.path.join(output_dir, f"analysis_enhanced.txt")
    with open(enhanced_file, 'w') as f:
        f.write("=== ENHANCED DETECTION RESULTS ===\n\n")
        f.write("Detected Objects:\n")
        for i, obj in enumerate(objects):
            f.write(f"Object {i+1}:\n")
            f.write(f"  Class: {obj['class']}\n")
            
            if obj['class'] in sam_bboxes:
                bbox = sam_bboxes[obj['class']]
                f.write(f"  Bounding Box (normalized): xmin={bbox['xmin']:.3f}, ymin={bbox['ymin']:.3f}, "
                       f"xmax={bbox['xmax']:.3f}, ymax={bbox['ymax']:.3f}\n")
                width = bbox['xmax'] - bbox['xmin']
                height = bbox['ymax'] - bbox['ymin']
                f.write(f"  Bounding Box (SLD format): [{bbox['xmin']:.3f}, {bbox['ymin']:.3f}, "
                       f"{width:.3f}, {height:.3f}]\n")
            else:
                bbox = obj['bbox']
                f.write(f"  Bounding Box (normalized): xmin={bbox[0]/img_width:.3f}, ymin={bbox[1]/img_height:.3f}, "
                       f"xmax={bbox[2]/img_width:.3f}, ymax={bbox[3]/img_height:.3f}\n")
            
            if obj['seg_points']:
                f.write("  Segmentation Points:\n")
                for point in obj['seg_points']:
                    f.write(f"    ({point[0]/img_width:.3f}, {point[1]/img_height:.3f})\n")
        
        scene_desc = re.search(r'Scene Description:\n(.*?)\n', content)
        spatial_rel = re.search(r'Spatial Relationships:\n(.*?)(?=\n=+|$)', content, re.DOTALL)
        
        f.write(f"\nScene Description:\n{scene_desc.group(1) if scene_desc else ''}\n")
        f.write(f"\nSpatial Relationships:\n{spatial_rel.group(1).strip() if spatial_rel else ''}\n")
        f.write("\n=====================\n")
