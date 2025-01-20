import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_affine_transformation_with_bbox_expansion(mask, transformation_matrix):
    """
    Applies a 3x3 affine transformation to the mask while expanding the bounding box to avoid cutting the object.

    Parameters:
    - mask: Input binary mask (numpy array).
    - transformation_matrix: 3x3 affine transformation matrix (numpy array).

    Returns:
    - Transformed mask (numpy array).
    """
    # Find the non-zero (object) region in the mask
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Original bounding box corners
    corners = np.array([
        [x_min, y_min, 1],
        [x_max, y_min, 1],
        [x_max, y_max, 1],
        [x_min, y_max, 1]
    ])

    # Transform the corners using the 3x3 matrix
    transformed_corners = np.dot(transformation_matrix, corners.T).T

    # Calculate the expanded bounding box
    new_x_min = int(np.floor(transformed_corners[:, 0].min()))
    new_y_min = int(np.floor(transformed_corners[:, 1].min()))
    new_x_max = int(np.ceil(transformed_corners[:, 0].max()))
    new_y_max = int(np.ceil(transformed_corners[:, 1].max()))

    # Ensure the bounding box fits within the mask dimensions
    height, width = mask.shape
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(width, new_x_max)
    new_y_max = min(height, new_y_max)

    # Define the size of the output
    expanded_width = new_x_max - new_x_min
    expanded_height = new_y_max - new_y_min

    # Warp the mask using the affine transformation
    transformed_mask = cv2.warpAffine(
        mask,
        transformation_matrix[:2],
        (expanded_width, expanded_height),
        flags=cv2.INTER_NEAREST
    )

    # Create an output mask with the same dimensions as the input
    output_mask = np.zeros_like(mask)
    output_mask[new_y_min:new_y_min + transformed_mask.shape[0], new_x_min:new_x_min + transformed_mask.shape[1]] = transformed_mask

    return output_mask

# Example usage
def main():
    # Example binary mask
    # Create a fake binary mask image
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 50, 255, -1)  # Draw a filled white circle

    # Define a 3x3 affine transformation matrix
    angle = 0  # degrees
    scale = 1.0
    tx, ty = 0.0, 0.0  # translation in pixels
    center = (mask.shape[1] // 2, mask.shape[0] // 2)

    # Construct the 3x3 affine transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    affine_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    affine_matrix[0, 2] += tx
    affine_matrix[1, 2] += ty

    # Apply the transformation
    transformed_mask = apply_affine_transformation_with_bbox_expansion(mask, affine_matrix)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Mask")
    plt.imshow(mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Transformed Mask with Expanded BBox")
    plt.imshow(transformed_mask, cmap='gray')
    plt.savefig('transformed_mask_3333.png')
    plt.close()

main()
