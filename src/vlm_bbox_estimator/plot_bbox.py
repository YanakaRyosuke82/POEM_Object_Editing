import cv2


def annotate_image_with_bbox(image, normalized_bbox):
    """
    Annotates the input image with a bounding box.

    Parameters:
    - image: The input image (numpy array)
    - normalized_bbox: List [xmin, ymin, xmax, ymax] with normalized coordinates (0-1)
    """
    height, width = image.shape[:2]
    
    # Convert normalized coordinates to absolute coordinates
    xmin = int(normalized_bbox[0] * width)
    ymin = int(normalized_bbox[1] * height)
    xmax = int(normalized_bbox[2] * width)
    ymax = int(normalized_bbox[3] * height)
    
    # Draw the rectangle on the image
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green color in BGR
    return image


# Example usage
image_path = "/dtu/blackhole/14/189044/marscho/Mask2Mask/508272.jpg"  # Path to the input image
output_path = "/dtu/blackhole/14/189044/marscho/Mask2Mask/annotated_image.jpg"  # Path to save the annotated image
normalized_bboxes = [
    [0.17, 0.15, 0.48, 0.84],
    [0.57, 0.38, 0.88, 0.83]
]  # Multiple bounding boxes

# Load the image from disk
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Image not found or unable to load at path: {image_path}")

# Resize image to 512x512
image = cv2.resize(image, (512, 512))

# Annotate the image with the bounding boxes
for bbox in normalized_bboxes:
    image = annotate_image_with_bbox(image, bbox)

# Save the annotated image to disk
cv2.imwrite(output_path, image)
