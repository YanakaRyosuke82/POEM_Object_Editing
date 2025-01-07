import cv2


def resize_and_crop_image(image_path, output_path, target_size=(512, 512)):
    """
    Resizes the input image to fit within the target size while maintaining the aspect ratio,
    and crops it to the target size of 512x512 pixels.

    Parameters:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the resized and cropped image.
    - target_size (tuple): Desired output size (width, height), should be (512, 512).
    """
    # Load the image from disk
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load at path: {image_path}")

    # Get the original dimensions
    height, width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = width / height
    target_width, target_height = target_size

    # Determine new dimensions while maintaining aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:  # Taller than wide or square
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image to fit the target size, maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate cropping coordinates to center the image to 512x512
    crop_start_x = max((new_width - target_width) // 2, 0)
    crop_start_y = max((new_height - target_height) // 2, 0)
    crop_end_x = crop_start_x + target_width
    crop_end_y = crop_start_y + target_height

    # Ensure dimensions are within bounds before cropping
    cropped_image = resized_image[crop_start_y : min(crop_end_y, new_height), crop_start_x : min(crop_end_x, new_width)]

    # Ensure the cropped image is exactly 512x512
    cropped_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Save the resized and cropped image to disk
    success = cv2.imwrite(output_path, cropped_image)
    if not success:
        raise IOError(f"Failed to save image at path: {output_path}")

    print(f"Image resized and cropped successfully: {output_path}")


# Example usage
if __name__ == "__main__":
    image_path = "/dtu/blackhole/14/189044/marscho/Mask2Mask/508272.jpg"  # Path to the input image
    output_path = "/dtu/blackhole/14/189044/marscho/Mask2Mask/resized_image.jpg"  # Path to save the resized image
    resize_and_crop_image(image_path, output_path, target_size=(512, 512))
