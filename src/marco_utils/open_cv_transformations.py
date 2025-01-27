import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json

def process_transformation_matrix_top_left(matrix, scaling_factor_y, bbox_ymin, bbox_height):
    """
    Process a transformation matrix to include scaling with feet grounding
    (assuming top-left origin).

    Parameters:
        matrix (np.array): The original transformation matrix.
        scaling_factor_y (float): The scaling factor in the Y-axis.
        bbox_ymin (float): The ymin (top) of the bounding box in pixels.
        bbox_height (float): The height of the bounding box in pixels.

    Returns:
        np.array: The adjusted transformation matrix.
    """
    # Original height and scaled height
    original_height = bbox_height
    new_height = scaling_factor_y * original_height

    # Bottom (ymax) position remains fixed
    bbox_ymax = bbox_ymin + original_height
    new_bbox_ymin = bbox_ymax - new_height  # Adjust ymin to keep ymax fixed

    # Translation correction to shift ymin
    translation_correction = new_bbox_ymin - bbox_ymin

    # Create a scaling and translation matrix
    scaling_translation_matrix = np.array([
        [1, 0, 0],                           # X-axis scaling remains unchanged
        [0, scaling_factor_y, translation_correction],  # Apply scaling and translation in Y-axis
        [0, 0, 1]
    ], dtype=np.float32)

    # Combine the original matrix with the new scaling and translation matrix
    adjusted_matrix = matrix @ scaling_translation_matrix
    return adjusted_matrix

def run_open_cv_transformations(matrix_transform_file, output_dir, MASK_FILE_NAME, ENHANCED_FILE_DESCRIPTION):

    # Load the transformation matrix
    loaded_matrix = np.load(matrix_transform_file)
    
    # Load and validate binary mask
    mask_path = os.path.join(output_dir, MASK_FILE_NAME)
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if binary_mask is None:
        raise FileNotFoundError(f"Could not load binary mask from {mask_path}")
    
    height, width = binary_mask.shape
    
    # Read object boundaries from detection file
    with open(ENHANCED_FILE_DESCRIPTION, 'r') as f:
        detection_text = f.read()

    # Extract bounding box coordinates
    bbox_line = [line for line in detection_text.split('\n') if 'Bounding Box' in line][0]
    ymin = float(bbox_line.split('ymin=')[1].split(',')[0])
    bbox_height = height * (float(bbox_line.split('ymax=')[1].split(')')[0]) - ymin)
    ymin = ymin * height

    # Process the transformation matrix to keep feet grounded
    loaded_matrix = process_transformation_matrix_top_left(loaded_matrix, 0.88, ymin, bbox_height)
    
    # Create mask transformation matrix by inverting y-scaling and y-translation
    mask_matrix = loaded_matrix.copy()
    mask_matrix[1,1] = 1/loaded_matrix[1,1]  # Invert y-scaling
    mask_matrix[1,2] = -loaded_matrix[1,2]   # Invert y-translation
    
    print("Loaded transformation matrix:")
    print(loaded_matrix)
    print("\nMask transformation matrix (y-scaling and translation inverted):")
    print(mask_matrix)

    print("\nBinary Mask Shape:", binary_mask.shape)

    # Get mask dimensions
    height, width = binary_mask.shape

    # Read object boundaries from detection file
    with open(ENHANCED_FILE_DESCRIPTION, 'r') as f:
        detection_text = f.read()

    # Extract bounding box coordinates
    bbox_line = [line for line in detection_text.split('\n') if 'Bounding Box' in line][0]
    xmin = float(bbox_line.split('xmin=')[1].split(',')[0])
    ymin = float(bbox_line.split('ymin=')[1].split(',')[0])
    xmax = float(bbox_line.split('xmax=')[1].split(',')[0])
    ymax = float(bbox_line.split('ymax=')[1].split(')')[0])

    # Convert normalized coordinates to pixel coordinates
    x_min = int(xmin * width)
    x_max = int(xmax * width)
    y_min = int(ymin * height)
    y_max = int(ymax * height)

    # Calculate object center
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Create translation matrices for centering
    # Note: We flip the y-translation since image coordinates have origin at top-left
    T_to_origin = np.array([
        [1, 0, -x_center],
        [0, 1, -y_center],
        [0, 0, 1]
    ])

    T_from_origin = np.array([
        [1, 0, x_center],
        [0, 1, y_center],
        [0, 0, 1]
    ])

    # Compose final transformation with inverted y-scaling for mask
    final_transform = T_from_origin @ mask_matrix @ T_to_origin

    # Generate coordinate grid
    y, x = np.mgrid[0:height, 0:width]
    coords = np.stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())), axis=1).T

    # Apply transformation to coordinates
    transformed_coords = final_transform @ coords
    transformed_coords /= transformed_coords[2]  # Normalize homogeneous coordinates

    # Reshape back to image dimensions
    transformed_x = transformed_coords[0].reshape(height, width)
    transformed_y = transformed_coords[1].reshape(height, width)

    # Remap the mask using transformed coordinates
    transformed_mask = cv2.remap(binary_mask,
                            transformed_x.astype(np.float32),
                            transformed_y.astype(np.float32),
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)

    # Transform bbox from enhanced detection file
    with open(ENHANCED_FILE_DESCRIPTION, 'r') as f:
        enhanced_text = f.read()

    # Extract enhanced bounding box coordinates
    enhanced_bbox_line = [line for line in enhanced_text.split('\n') if 'Bounding Box' in line][0]
    enhanced_xmin = float(enhanced_bbox_line.split('xmin=')[1].split(',')[0])
    enhanced_ymin = float(enhanced_bbox_line.split('ymin=')[1].split(',')[0])
    enhanced_xmax = float(enhanced_bbox_line.split('xmax=')[1].split(',')[0])
    enhanced_ymax = float(enhanced_bbox_line.split('ymax=')[1].split(',')[0])

    # Convert normalized coordinates to pixel coordinates
    bbox_coords = np.array([
        [enhanced_xmin * width, enhanced_ymin * height, 1],
        [enhanced_xmax * width, enhanced_ymin * height, 1],
        [enhanced_xmax * width, enhanced_ymax * height, 1],
        [enhanced_xmin * width, enhanced_ymax * height, 1]
    ]).T

    # Transform bbox coordinates with original transformation (not inverted)
    transformed_bbox = T_from_origin @ loaded_matrix @ T_to_origin @ bbox_coords
    transformed_bbox /= transformed_bbox[2]  # Normalize homogeneous coordinates

    # Convert back to normalized coordinates
    transformed_bbox_normalized = np.array([
        transformed_bbox[0] / width,
        transformed_bbox[1] / height
    ])

    # Print transformed bbox to console in both formats
    print("\nTransformed Bounding Box (normalized):")
    print(f"xmin={transformed_bbox_normalized[0,0]:.3f}, "
        f"ymin={transformed_bbox_normalized[1,0]:.3f}, "
        f"xmax={transformed_bbox_normalized[0,1]:.3f}, "
        f"ymax={transformed_bbox_normalized[1,2]:.3f}")

    # Convert to SLD format [Top-left x, Top-left y, Width, Height]
    sld_x = transformed_bbox_normalized[0,0]
    sld_y = transformed_bbox_normalized[1,0]
    sld_width = transformed_bbox_normalized[0,1] - transformed_bbox_normalized[0,0]
    sld_height = transformed_bbox_normalized[1,2] - transformed_bbox_normalized[1,0]

    # Store original and transformed bboxes in SLD format
    bbox_data = {
        "original_bbox": [enhanced_xmin, enhanced_ymin, enhanced_xmax - enhanced_xmin, enhanced_ymax - enhanced_ymin],
        "transformed_bbox": [sld_x, sld_y, sld_width, sld_height]
    }
     # Save bbox data in SLD format
    bbox_sld_file = os.path.join(output_dir, 'bbox_sld.json')
    with open(bbox_sld_file, 'w') as f:
        json.dump(bbox_data, f, indent=2)

    # Save transformed bbox to file (both formats)
    transformed_bbox_file = os.path.join(output_dir, 'transformed_bbox.txt')
    with open(transformed_bbox_file, 'w') as f:
        f.write("Transformed Bounding Box (normalized):\n")
        f.write(f"xmin={transformed_bbox_normalized[0,0]:.3f}, ")
        f.write(f"ymin={transformed_bbox_normalized[1,0]:.3f}, ")
        f.write(f"xmax={transformed_bbox_normalized[0,1]:.3f}, ")
        f.write(f"ymax={transformed_bbox_normalized[1,2]:.3f}\n")
        f.write("\nTransformed Bounding Box (SLD format):\n")
        f.write(f"[{sld_x:.3f}, {sld_y:.3f}, {sld_width:.3f}, {sld_height:.3f}]\n")

    print(f"\nTransformed bbox saved to: {transformed_bbox_file}")

    # Save the transformed mask
    output_mask_path = os.path.join(output_dir, 'transformed_mask.png')
    cv2.imwrite(output_mask_path, transformed_mask)
    print(f"\nTransformed mask saved to: {output_mask_path}")

    # Save the original and transformed masks as .npy files
    source_mask_path = os.path.join(output_dir, 'source_mask.npy')
    target_mask_path = os.path.join(output_dir, 'target_mask.npy')
    np.save(source_mask_path, binary_mask)
    np.save(target_mask_path, transformed_mask)
    print(f"\nOriginal mask saved to: {source_mask_path}")
    print(f"Transformed mask saved to: {target_mask_path}")
    # Visualization - Mask Transformation
    plt.figure(figsize=(15, 5))

    # Create subplots for better comparison
    plt.subplot(131)
    # Original mask visualization
    plt.imshow(np.zeros((height, width, 3), dtype=np.uint8) + 255)  # White background
    mask_overlay = np.zeros((height, width, 4))
    mask_overlay[binary_mask > 0] = [0, 0.47, 1, 0.8]  # Professional blue with alpha
    plt.imshow(mask_overlay)
    plt.title('Original Mask', fontsize=22, pad=10)  # Increased from 14
    # plt.plot([0, width-1, width-1, 0, 0], [0, 0, height-1, height-1, 0], 'k-', linewidth=1)  # Add corners
    plt.axis('off')

    plt.subplot(132) 
    # Transformed mask visualization
    plt.imshow(np.zeros((height, width, 3), dtype=np.uint8) + 255)  # White background
    mask_overlay = np.zeros((height, width, 4))
    mask_overlay[transformed_mask > 0] = [1, 0.2, 0.2, 0.8]  # Professional red with alpha
    plt.imshow(mask_overlay)
    plt.title('Transformed Mask', fontsize=22, pad=10)  # Increased from 14
    # plt.plot([0, width-1, width-1, 0, 0], [0, 0, height-1, height-1, 0], 'k-', linewidth=1)  # Add corners
    plt.axis('off')

    plt.subplot(133)
    # Overlay comparison
    plt.imshow(np.zeros((height, width, 3), dtype=np.uint8) + 255)  # White background
    # Original mask in blue
    mask_overlay = np.zeros((height, width, 4))
    mask_overlay[binary_mask > 0] = [0, 0.47, 1, 0.5]  # Semi-transparent blue
    plt.imshow(mask_overlay)
    # Transformed mask in red
    mask_overlay = np.zeros((height, width, 4))
    mask_overlay[transformed_mask > 0] = [1, 0.2, 0.2, 0.5]  # Semi-transparent red
    plt.imshow(mask_overlay)
    plt.title('Overlay', fontsize=22, pad=10)  # Increased from 14
    # plt.plot([0, width-1, width-1, 0, 0], [0, 0, height-1, height-1, 0], 'k-', linewidth=1)  # Add corners
    plt.axis('off')

    # Add overall title and adjust layout
    # plt.suptitle('Mask Transformation Analysis', fontsize=18, y=1.05)  # Increased from 16
    plt.tight_layout()

    # Matrix text commented out but preserved
    # plt.text(0.02, 0.98, f"Original Matrix:\n{loaded_matrix}\n\nMask Matrix:\n{mask_matrix}", 
    #          transform=plt.gca().transAxes,
    #          fontsize=12,  # Increased from 10
    #          color='red', 
    #          verticalalignment='top',
    #          fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transformation_vis.png'), 
                transparent=False,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()

    # Visualization - Bounding Boxes
    plt.figure(figsize=(15, 10))
    plt.gca().set_aspect('equal')

    # Plot original bbox with enhanced style
    original_bbox = plt.Rectangle((enhanced_xmin * width, enhanced_ymin * height),
                                (enhanced_xmax - enhanced_xmin) * width,
                                (enhanced_ymax - enhanced_ymin) * height,
                                fill=False, 
                                color='#0066CC',  # Refined blue
                                linewidth=3, 
                                linestyle='--',
                                label='Initial Position')

    # Plot transformed bbox with enhanced style
    transformed_bbox_plot = plt.Polygon(transformed_bbox[:2].T, 
                                    fill=False, 
                                    color='#CC3300',  # Refined red
                                    linewidth=3,
                                    linestyle='-',
                                    label='Transformed Position')

    plt.gca().add_patch(original_bbox)
    plt.gca().add_patch(transformed_bbox_plot)

    plt.xlim(0, width)
    plt.ylim(height, 0)  # Flip y-axis to match image coordinates
    plt.grid(False)  # Remove grid for cleaner look
    plt.legend(loc='upper right', 
            frameon=True,
            framealpha=0.9,
            edgecolor='none',
            fontsize=22)  # Default size is ~10, so 5x larger is 50
    plt.savefig(os.path.join(output_dir, 'bbox_transformation_vis.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.show()
