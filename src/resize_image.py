from PIL import Image
import io


FILE_PATH = "demo/self_correction/src_image_marco/elephant.png"

# Load the image from the file path
img = Image.open(FILE_PATH)

# Resize the image to 512x512
img = img.resize((512, 512))

# Save the resized image as png in the same folder
img.save("demo/self_correction/src_image_marco/elephant_resized.png")
