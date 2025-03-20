import cv2
import numpy as np

# Load the image
image = cv2.imread("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/exp_chess/input/sample_001/input_image.png")

# Create a black background of size 512x512
background = np.zeros((512, 512, 3), dtype=np.uint8)

# Get image dimensions
height, width = image.shape[:2]

# Calculate crop dimensions
if width > height:
    # Image is wider than tall
    crop_size = height
    x_start = (width - height) // 2
    y_start = 0
else:
    # Image is taller than wide
    crop_size = width
    x_start = 0
    y_start = (height - width) // 2

# Crop the image to a square
cropped_image = image[y_start : y_start + crop_size, x_start : x_start + crop_size]

# Resize the cropped square image to 512x512
resized_image = cv2.resize(cropped_image, (512, 512))

# Place the resized image onto the black background
background = resized_image

# Save the final image
cv2.imwrite("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/exp_chess/input/sample_001/input_image2.png", background)
