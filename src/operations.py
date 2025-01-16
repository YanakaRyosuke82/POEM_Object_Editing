

import torch

from utils_marco import generate_image_from_prompt, get_mask_from_sam_detector





def add_operation(original_latents: torch.Tensor, operation_params: dict) -> torch.Tensor:
    """Add operation to latent representation.
    1. pregenarate an object
    2. grounding sam operates on the image space and gets of a binary mask 
    3. move masked latetns from the pregenerated latetnts to the original-image lantets the mask to the latents
    """

    pregen_img, pregen_latent_history = generate_image_from_prompt(operation_params["prompt"])
    pregen_img.save("pregen_img.png")

    # # 2. grounding sam operates on the image space and gets of a binary mask 
    # pregen_mask = get_mask_from_sam_detector(pregen_img)



def remove_operation(latents: torch.Tensor, operation_params: dict) -> torch.Tensor:
    """Remove operation from latent representation."""
    return None



def modify_operation(latents: torch.Tensor, operation_params: dict) -> torch.Tensor:
    """Modify operation in latent representation."""
    return None
