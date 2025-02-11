import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def _process_object(mask: np.array, transformation_matrix: np.array) -> np.array:
    # Find object boundaries
    where_filter = np.where(mask != 0)
    y_min, x_min = np.min(where_filter, axis=1)
    y_max, x_max = np.max(where_filter, axis=1)
    x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

    T_move_current_center_to_origin = np.array(
        [
            [1, 0, -x_center],
            [0, 1, -y_center],
            [0, 0, 1],
        ]
    )
    T_move_origin_to_current_center = np.array(
        [
            [1, 0, x_center],
            [0, 1, y_center],
            [0, 0, 1],
        ]
    )
    T_wrt_image_center = T_move_origin_to_current_center @ transformation_matrix @ T_move_current_center_to_origin

    # Apply the transformation to the object
    transformed_mask = cv2.warpAffine(
        mask,
        T_wrt_image_center[:2],
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    return transformed_mask


def run_open_cv_transformations(matrix_transform_file, output_dir, oracle_mask_path, ENHANCED_FILE_DESCRIPTION):

    # Get object ID from file
    object_id_path = os.path.join(output_dir, "object_id.txt")
    with open(object_id_path, "r") as f:
        object_id = int(f.read().strip())

    # Load and validate binary mask
    mask_file_name = f"mask_{object_id}.png"
    mask_path = os.path.join(output_dir, mask_file_name)
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if binary_mask is None:
        raise FileNotFoundError(f"Could not load binary mask from {mask_path}")

    # Load oracle mask
    oracle_mask = cv2.imread(oracle_mask_path, cv2.IMREAD_GRAYSCALE)
    if oracle_mask is None:
        raise FileNotFoundError(f"Could not load oracle mask from {oracle_mask_path}")

    # Load the transformation matrix
    loaded_matrix = np.load(matrix_transform_file)

    height, width = binary_mask.shape

    # Find original object  boundaries
    where_filter = np.where(binary_mask != 0)
    y_min, x_min = np.min(where_filter, axis=1)
    y_max, x_max = np.max(where_filter, axis=1)
    older_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Apply the transformation matrix to both masks
    transformed_mask = _process_object(binary_mask, loaded_matrix)
    transformed_oracle = _process_object(oracle_mask, loaded_matrix)

    # Find transformed object boundaries
    where_filter = np.where(transformed_mask != 0)
    y_min, x_min = np.min(where_filter, axis=1)
    y_max, x_max = np.max(where_filter, axis=1)
    new_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Calculate normalized bbox
    normalized_bbox = [x_min / width, y_min / height, (x_max - x_min) / width, (y_max - y_min) / height]

    # Calculate normalized transformed bbox
    normalized_transformed_bbox = [new_bbox[0] / width, new_bbox[1] / height, new_bbox[2] / width, new_bbox[3] / height]

    # Read existing analysis file
    with open(ENHANCED_FILE_DESCRIPTION, "r") as f:
        analysis_lines = f.readlines()

    # Find the object entry and add bbox
    for i, line in enumerate(analysis_lines):
        if "Bounding Box (SLD format):" in line:
            bbox_line_transformed = f"  Bounding Box (SLD format) transformed: {normalized_transformed_bbox}\n"
            analysis_lines.insert(i + 1, bbox_line_transformed)
            break

    # Write back updated analysis
    with open(ENHANCED_FILE_DESCRIPTION, "w") as f:
        f.writelines(analysis_lines)

    #### plotting
    # Save the transformed masks
    output_mask_path = os.path.join(output_dir, "transformed_mask.png")
    output_oracle_path = os.path.join(output_dir, "transformed_oracle.png")
    cv2.imwrite(output_mask_path, transformed_mask)
    cv2.imwrite(output_oracle_path, transformed_oracle)
    print(f"\nTransformed mask saved to: {output_mask_path}")
    print(f"Transformed oracle saved to: {output_oracle_path}")

    # Save the original and transformed masks as .npy files
    source_mask_path = os.path.join(output_dir, "source_mask.npy")
    target_mask_path = os.path.join(output_dir, "target_mask.npy")
    source_oracle_path = os.path.join(output_dir, "source_oracle.npy")
    target_oracle_path = os.path.join(output_dir, "target_oracle.npy")
    np.save(source_mask_path, binary_mask)
    np.save(target_mask_path, transformed_mask)
    np.save(source_oracle_path, oracle_mask)
    np.save(target_oracle_path, transformed_oracle)
    print(f"\nOriginal mask saved to: {source_mask_path}")
    print(f"Transformed mask saved to: {target_mask_path}")
    print(f"Original oracle saved to: {source_oracle_path}")
    print(f"Transformed oracle saved to: {target_oracle_path}")
    # Visualization - Mask Transformation
    plt.figure(figsize=(15, 5))

    # Create subplots for better comparison
    plt.subplot(131)
    # Original mask visualization
    plt.imshow(np.zeros((height, width, 3), dtype=np.uint8) + 255)  # White background
    mask_overlay = np.zeros((height, width, 4))
    mask_overlay[binary_mask > 0] = [0, 0.47, 1, 0.8]  # Professional blue with alpha
    plt.imshow(mask_overlay)
    plt.title("Original Mask", fontsize=22, pad=10)  # Increased from 14
    plt.plot([0, width - 1, width - 1, 0, 0], [0, 0, height - 1, height - 1, 0], "k-", linewidth=1)  # Add corners
    plt.axis("off")

    plt.subplot(132)
    # Transformed mask visualization
    plt.imshow(np.zeros((height, width, 3), dtype=np.uint8) + 255)  # White background
    mask_overlay = np.zeros((height, width, 4))
    mask_overlay[transformed_mask > 0] = [1, 0.2, 0.2, 0.8]  # Professional red with alpha
    plt.imshow(mask_overlay)
    plt.title("Transformed Mask", fontsize=22, pad=10)  # Increased from 14
    plt.plot([0, width - 1, width - 1, 0, 0], [0, 0, height - 1, height - 1, 0], "k-", linewidth=1)  # Add corners
    plt.axis("off")

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
    plt.title("Overlay", fontsize=22, pad=10)  # Increased from 14
    plt.plot([0, width - 1, width - 1, 0, 0], [0, 0, height - 1, height - 1, 0], "k-", linewidth=1)  # Add corners
    plt.axis("off")

    # Add overall title and adjust layout
    # plt.suptitle('Mask Transformation Analysis', fontsize=18, y=1.05)  # Increased from 16
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transformation_vis.png"), transparent=False, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()

    # Visualization - Bounding Boxes
    plt.figure(figsize=(15, 10))
    plt.gca().set_aspect("equal")

    # Plot original bbox with enhanced style
    original_bbox = plt.Rectangle(
        (older_bbox[0], older_bbox[1]),
        older_bbox[2],
        older_bbox[3],
        fill=False,
        color="#0066CC",  # Refined blue
        linewidth=3,
        linestyle="--",
        label="Initial Position",
    )

    # Plot transformed bbox with enhanced style
    transformed_bbox = plt.Rectangle(
        (new_bbox[0], new_bbox[1]),
        new_bbox[2],
        new_bbox[3],
        fill=False,
        color="#CC3300",  # Refined red
        linewidth=3,
        linestyle="-",
        label="Transformed Position",
    )

    plt.gca().add_patch(original_bbox)
    plt.gca().add_patch(transformed_bbox)

    plt.xlim(0, width)
    plt.ylim(height, 0)  # Flip y-axis to match image coordinates
    plt.grid(False)  # Remove grid for cleaner look
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=22)  # Default size is ~10, so 5x larger is 50
    plt.savefig(os.path.join(output_dir, "bbox_transformation_vis.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    return transformed_mask, transformed_oracle
