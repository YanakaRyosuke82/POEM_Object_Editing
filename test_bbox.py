import cv2
import numpy as np

# Define image dimensions
image_width, image_height = 512, 512

# Create a black image
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Bounding box annotations
annotations = [
    ["dog #1", [0.003, 0.209, 0.444, 0.411]],  # Width and Height derived from x2-x1, y2-y1
    ["cat #1", [0.457, 0.342, 0.441, 0.253]]
]

# Draw bounding boxes and annotations on the image
for label, bbox in annotations:
    # Convert relative bbox coordinates to absolute pixel values
    x_min = int(bbox[0] * image_width)
    y_min = int(bbox[1] * image_height)
    x_max = int((bbox[0] + bbox[2]) * image_width)
    y_max = int((bbox[1] + bbox[3]) * image_height)

    # Draw rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Put label text above the rectangle
    text_position = (x_min, y_min - 10 if y_min > 20 else y_min + 20)
    cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Save the image to disk
output_path = 'annotated_image.png'
cv2.imwrite(output_path, image)

print(f"Annotated image saved to {output_path}")