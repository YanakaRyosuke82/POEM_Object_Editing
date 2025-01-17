import cv2
import numpy as np

def apply_affine_transformation(image, bbox, transformation_matrix):
    """
    Applies any 3x3 affine transformation to the content inside a normalized bounding box.

    Parameters:
    - image: Input image (numpy array).
    - bbox: Normalized bounding box in the format [Top-left x, Top-left y, Width, Height].
    - transformation_matrix: 3x3 affine transformation matrix (numpy array).
    
    Returns:
    - Transformed image with the affine transformation applied to the bbox area.
    """
    # Get image dimensions
    img_h, img_w = image.shape[:2]

    # Denormalize the bounding box coordinates
    x = int(bbox[0] * img_w)
    y = int(bbox[1] * img_h)
    w = int(bbox[2] * img_w)
    h = int(bbox[3] * img_h)

    # Extract the region of interest (ROI)
    roi = image[y:y+h, x:x+w]

    # Calculate center of ROI
    roi_center_x = w // 2
    roi_center_y = h // 2

    # Create translation matrices
    translation_to_origin = np.array([
        [1, 0, -roi_center_x],
        [0, 1, -roi_center_y],
        [0, 0, 1]
    ])
    translation_back = np.array([
        [1, 0, roi_center_x],
        [0, 1, roi_center_y],
        [0, 0, 1]
    ])

    # Combine transformations: translate to origin -> transform -> translate back
    final_transform = translation_back @ transformation_matrix @ translation_to_origin

    # Perform the affine transformation on the ROI
    transformed_roi = cv2.warpPerspective(roi, final_transform, (w, h), flags=cv2.INTER_LINEAR)

    # Create output image
    transformed_image = image.copy()
    transformed_image[y:y+h, x:x+w] = transformed_roi

    return transformed_image

# Example usage:
if __name__ == "__main__":
    # Create a dummy image
    image_width = 512
    image_height = 512
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
    # Add some test content - a colored rectangle
    cv2.rectangle(image, (100, 100), (400, 400), (0, 255, 0), -1)
    cv2.rectangle(image, (200, 200), (300, 300), (0, 0, 255), -1)

    # Define the normalized bounding box [Top-left x, Top-left y, Width, Height]
    bbox = [0.1, 0.1, 0.4, 0.4]  # Example normalized bbox
    # Define a 3x3 affine transformation matrix for 45 degree rotation
    transformation_matrix = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
        [0, 0, 1]
    ])

    # Apply the affine transformation
    transformed_image = apply_affine_transformation(image, bbox, transformation_matrix)

    # Save the result to disk
    output_path = "transformed_image.png"
    cv2.imwrite(output_path, transformed_image)
    print(f"Saved transformed image to {output_path}")
