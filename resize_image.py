from PIL import Image
import io


def resize_image(input_path: str, output_path: str, target_size: tuple = (512, 512)) -> None:
    """
    Resize an image to a target size and save it.
    
    Args:
        input_path: Path to the input image file
        output_path: Path where the resized image will be saved
        target_size: Tuple of (width, height) for the target size, defaults to (512, 512)
    """
    # Load the image from the file path
    img = Image.open(input_path)

    # Resize the image to target size
    img = img.resize(target_size)

    # Save the resized image
    img.save(output_path)
