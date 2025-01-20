import cv2
import numpy as np
import matplotlib.pyplot as plt

def define_affine_transformation(rotation_angle, translation, scaling):
    """
    Creates a 3x3 affine transformation matrix based on user inputs.

    Parameters:
    - rotation_angle (float): Rotation angle in degrees.
    - translation (tuple): Translation as (tx, ty).
    - scaling (tuple): Scaling as (sx, sy).  # scaling in x, y direction

    Returns:
    - np.ndarray: A 3x3 affine transformation matrix.
    """
    # Convert angle from degrees to radians
    theta = np.radians(rotation_angle)

    # Decompose inputs
    tx, ty = translation
    sx, sy = scaling

    # Define the affine matrix
    affine_matrix = np.array([
        [sx * np.cos(theta), -sy * np.sin(theta), tx],
        [sx * np.sin(theta),  sy * np.cos(theta), ty],
        [0,                  0,                  1]
    ], dtype=np.float32)
    return affine_matrix.astype(np.float32)



def apply_affine_transform(image, bbox, M_affine):
    """
    Applies a 3x3 affine transformation matrix to a specified region of interest (ROI) within an image.

    This function takes an input image and a bounding box that defines the ROI. It applies the given affine 
    transformation matrix to this region, effectively transforming the content within the bounding box. The 
    transformation includes operations such as rotation, scaling, and translation. The transformed region is 
    then reintegrated into the original image, replacing the original content.

    Parameters:
    - image (np.ndarray): The input image on which the transformation is to be applied. It should be a 
      multi-dimensional array representing the image in a format compatible with OpenCV.
    - bbox (tuple): A tuple containing four integers that define the bounding box of the ROI in the format 
      (x_min, x_max, y_min, y_max). These coordinates specify the top-left and bottom-right corners of the 
      rectangle enclosing the ROI.
    - M_affine (np.ndarray): A 3x3 affine transformation matrix. This matrix defines the transformation to 
      be applied to the ROI, including any combination of translation, rotation, and scaling.

    Returns:
    - np.ndarray: The image with the affine transformation applied to the specified ROI. The transformed 
      region is seamlessly integrated back into the original image, maintaining the overall image dimensions.

    The function follows these steps:
    1. Extracts the ROI from the input image based on the provided bounding box.
    2. Creates a transparent image and a mask to isolate the ROI.
    3. Translates the ROI and mask to the origin to facilitate transformation.
    4. Applies the affine transformation to the translated ROI.
    5. Translates the transformed ROI back to its original position.
    6. Integrates the transformed ROI back into the original image, ensuring that the rest of the image 
       remains unchanged.
    """

    # Unpack bbox in new order: (x_min, x_max, y_min, y_max)
    x_min, x_max, y_min, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    # Create a transparent image of the same size as the original
    transparent_image = np.zeros_like(image, dtype=np.float32)

    # Copy the ROI to the transparent image
    transparent_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

    # Create a mask for the transformed ROI
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1.0

    # Translate the mask and transparent image to the origin
    translation_to_origin = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    # Translate back to the original position after transformation
    translation_back = np.array([
        [1, 0, x_min],
        [0, 1, y_min],
        [0, 0, 1]
    ])

    # Combine transformations: translate to origin, apply affine, then translate back
    combined_transform = translation_back @ M_affine @ translation_to_origin

    # Apply the combined transformation to the transparent image
    transformed_image = cv2.warpAffine(
        transparent_image,
        combined_transform[:2],
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # Apply the same combined transformation to the mask
    transformed_mask = cv2.warpAffine(
        mask,
        combined_transform[:2],
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST
    )


    ###### Quickly get the bounding box of the translated object in the transparent image
    y_indices, x_indices = np.where(transformed_mask)
    if y_indices.size > 0 and x_indices.size > 0:
        new_x_min, new_x_max = int(x_indices.min()), int(x_indices.max())
        new_y_min, new_y_max = int(y_indices.min()), int(y_indices.max())
    else:
        new_x_min, new_x_max, new_y_min, new_y_max = 0, 0, 0, 0
    new_bbox = [new_x_min, new_x_max, new_y_min, new_y_max]
    #########################################################

    # Remove the original ROI from the image
    image[y_min:y_max, x_min:x_max] = np.random.normal(loc=0.0, scale=1.0, size=image[y_min:y_max, x_min:x_max].shape)

    # Combine the transformed image with the original image
    combined_image = image.copy()
    combined_image[transformed_mask.astype(bool)] = transformed_image[transformed_mask.astype(bool)]

    return combined_image, new_bbox
    # return combined_image, None


def main():
    # Create a 512x512 image with a white square for demonstration and yellow background
    image = np.zeros((512, 512, 3), dtype=np.float32)
    image[:, :] = [0, 255, 255]  # Fill with yellow
    cv2.rectangle(image, (150, 150), (350, 350), (255, 255, 255), -1)

    # Define the bounding box of the ROI
    # bbox format: (x_min, x_max, y_min, y_max) where:
    # x_min: x-coordinate of left edge
    # x_max: x-coordinate of right edge
    # y_min: y-coordinate of top edge
    # y_max: y-coordinate of bottom edge
    bbox = (150, 350, 150, 350)  # Changed order to match new format

    # Define an affine transformation matrix (rotation + scaling)
    angle =5.0  # Rotation angle in degrees
    scale = (1.0, 1.2)  # Uniform scaling
    translation = (25.0, +50)  # No translation
    M_affine = define_affine_transformation(angle, translation, scale)

    # Apply the affine transformation
    transformed_image, _ = apply_affine_transform(image, bbox, M_affine)

    # Debug plot to visualize transformed image with the old bounding box
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot transformed image with the old bounding box
    ax.imshow(cv2.cvtColor(transformed_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    rect = plt.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2], edgecolor='blue', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    ax.set_title("Transformed Image with Old Bounding Box")

    # Plot origin point with yellow circle and coordinate axes
    origin_x = 0  # Origin x coordinate
    origin_y = 0  # Origin y coordinate 

    # Plot origin circle
    circle = plt.Circle((origin_x, origin_y), radius=20, color='red', fill=True)
    ax.add_patch(circle)

    # Plot coordinate axes with arrows
    axis_length = 100
    ax.arrow(origin_x, origin_y, axis_length, 0, head_width=10, head_length=10, fc='red', ec='red')  # x-axis
    ax.arrow(origin_x, origin_y, 0, axis_length, head_width=10, head_length=10, fc='red', ec='red')  # y-axis
    ax.text(axis_length + 10, 0, 'x', fontsize=12, color='red')
    ax.text(0, axis_length + 10, 'y', fontsize=12, color='red')

    plt.savefig('affine_test.png')
    print("Saved image to affine_test.png")
    plt.close()

if __name__ == "__main__":
    main()