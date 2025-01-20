import torch
import numpy as np
from . import utils
from utils import torch_device
import matplotlib.pyplot as plt
from PIL import Image
import models
import torch.nn.functional as F
from torchvision import transforms as tvt
import os
import logging
import math
import cv2

from utils.affine_transformation_marco import apply_affine_transform, define_affine_transformation
# from models import pipelines, encode_prompts


def get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    # Obtain with torch.float32 and cast to float16 if needed
    # Directly obtaining latents in float16 will lead to different latents
    latents_base = torch.randn(
        (batch_size, in_channels, height // 8, width // 8),
        generator=generator,
        dtype=dtype,
    ).to(torch_device, dtype=dtype)

    return latents_base


def get_scaled_latents(
    batch_size, in_channels, height, width, generator, dtype, scheduler
):
    latents_base = get_unscaled_latents(
        batch_size, in_channels, height, width, generator, dtype
    )
    latents_base = latents_base * scheduler.init_noise_sigma
    return latents_base


def blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=0.01):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    assert not torch.allclose(
        latents_bg, latents_fg
    ), "latents_bg should be independent with latents_fg"

    dtype = latents_bg.dtype
    latents = (
        latents_bg * (1.0 - fg_mask)
        + (
            latents_bg * np.sqrt(1.0 - fg_blending_ratio)
            + latents_fg * np.sqrt(fg_blending_ratio)
        )
        * fg_mask
    )
    latents = latents.to(dtype=dtype)

    return latents


@torch.no_grad()
def compose_latents(
    model_dict,
    latents_all_list,
    mask_tensor_list,
    num_inference_steps,
    overall_batch_size,
    height,
    width,
    bg_seed=None,
    compose_box_to_bg=True,
    use_fast_schedule=False,
    fast_after_steps=None,
    latents_bg=None,
):
    unet, scheduler, dtype = model_dict.unet, model_dict.scheduler, model_dict.dtype

    generator = torch.manual_seed(
        bg_seed
    )  # Seed generator to create the inital latent noise
    latents_bg = get_scaled_latents(
        51,
        unet.config.in_channels,
        height,
        width,
        generator,
        dtype,
        scheduler,
    )
    latents_bg = latents_bg.unsqueeze(1)

    # Other than t=T (idx=0), we only have masked latents. This is to prevent accidentally loading from non-masked part. Use same mask as the one used to compose the latents.
    if use_fast_schedule:
        # If we use fast schedule, we only compose the frozen steps because the later steps do not match.
        composed_latents = torch.zeros(
            (fast_after_steps + 1, *latents_bg.shape), dtype=dtype
        )
    else:
        # Otherwise we compose all steps so that we don't need to compose again if we change the frozen steps.
        composed_latents = torch.zeros((latents_bg.shape), dtype=dtype)
        # composed_latents = latents_bg

    foreground_indices = torch.zeros(latents_bg.shape[-2:], dtype=torch.long)

    mask_size = np.array([mask_tensor.sum().item() for mask_tensor in mask_tensor_list])

    # Compose the largest mask first
    mask_order = np.argsort(-mask_size)

    existing_objects = torch.zeros(latents_bg.shape[-2:], dtype=torch.bool)
    # print(len(mask_order))
    # exit()
    for idx, mask_idx in enumerate(mask_order):
        latents_all, mask_tensor = (
            latents_all_list[mask_idx],
            mask_tensor_list[mask_idx],
        )

        mask_tensor_expanded = mask_tensor[None, None, None, ...].repeat(51, 1, 4, 1, 1)
        composed_latents[mask_tensor_expanded == 1] = latents_all[
            mask_tensor_expanded == 1
        ]
        existing_objects |= mask_tensor

    existing_objects_expanded = existing_objects[None, None, None, ...].repeat(
        51, 1, 4, 1, 1
    )
    composed_latents[existing_objects_expanded == 0] = latents_bg.cpu()[
        existing_objects_expanded == 0
    ]

    composed_latents, foreground_indices = composed_latents.to(
        torch_device
    ), existing_objects.to(torch_device)
    return composed_latents, foreground_indices


def align_with_bboxes(
    latents_all_list, mask_tensor_list, bboxes, horizontal_shift_only=False
):
    """
    Each offset in `offset_list` is `(x_offset, y_offset)` (normalized).
    """
    new_latents_all_list, new_mask_tensor_list, offset_list = [], [], []
    for latents_all, mask_tensor, bbox in zip(
        latents_all_list, mask_tensor_list, bboxes
    ):
        x_src_center, y_src_center = utils.binary_mask_to_center(
            mask_tensor, normalize=True
        )
        x_min_dest, y_min_dest, x_max_dest, y_max_dest = bbox
        x_dest_center, y_dest_center = (x_min_dest + x_max_dest) / 2, (
            y_min_dest + y_max_dest
        ) / 2
        # print("src (x,y):", x_src_center, y_src_center, "dest (x,y):", x_dest_center, y_dest_center)
        x_offset, y_offset = x_dest_center - x_src_center, y_dest_center - y_src_center
        if horizontal_shift_only:
            y_offset = 0.0
        offset = x_offset, y_offset
        latents_all = utils.shift_tensor(
            latents_all, x_offset, y_offset, offset_normalized=True
        )
        mask_tensor = utils.shift_tensor(
            mask_tensor, x_offset, y_offset, offset_normalized=True
        )
        new_latents_all_list.append(latents_all)
        new_mask_tensor_list.append(mask_tensor)
        offset_list.append(offset)

    return new_latents_all_list, new_mask_tensor_list, offset_list


def coord_transform(coords, width):
    """
    Transforms coordinates from normalized [Top-left x, Top-left y, Height, Width] format
    to pixel coordinates in [x_min, x_max, y_min, y_max] format.
    
    Args:
        coords: Normalized coordinates [x_min, y_min, height, width] in range [0,1]
        width: Width of the target image in pixels
        
    Returns:
        Tuple of integer pixel coordinates (x_min, x_max, y_min, y_max)
    """
    x_min, y_min, h, w = coords
    x_max = x_min + h  # Add height to get x_max
    y_max = y_min + w  # Add width to get y_max
    new_coords = (
        int(x_min * width),  # Scale x_min to pixels
        int(x_max * width),  # Scale x_max to pixels  
        int(y_min * width),  # Scale y_min to pixels
        int(y_max * width),  # Scale y_max to pixels
    )
    return new_coords


def apply_affine_transformation_gpt(image, bbox, transformation_matrix):
    """
    Applies any 3x3 affine transformation to the content inside a normalized bounding box.
    Handles cases where transformed content goes outside image boundaries by cropping.

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
    roi_original = image[y:y+h, x:x+w].copy()

    # Calculate center of original bbox
    center_x = x + w//2
    center_y = y + h//2

    # Create translation matrices to move center to origin and back
    to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    from_origin = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])

    # Compute the four corners of the bbox
    corners = np.array([
        [x, y, 1],
        [x + w, y, 1],
        [x + w, y + h, 1],
        [x, y + h, 1]
    ])

    # Transform corners: translate to origin -> transform -> translate back
    final_transform = from_origin @ transformation_matrix @ to_origin
    transformed_corners = np.dot(final_transform, corners.T).T

    # Convert homogeneous coordinates back to cartesian
    transformed_corners[:, 0] = transformed_corners[:, 0] / transformed_corners[:, 2]
    transformed_corners[:, 1] = transformed_corners[:, 1] / transformed_corners[:, 2]

    # Get bounds of transformed corners before cropping
    x_min_uncropped = np.floor(transformed_corners[:, 0].min()).astype(int)
    y_min_uncropped = np.floor(transformed_corners[:, 1].min()).astype(int)
    x_max_uncropped = np.ceil(transformed_corners[:, 0].max()).astype(int)
    y_max_uncropped = np.ceil(transformed_corners[:, 1].max()).astype(int)

    # Calculate size needed for full transformed image
    w_uncropped = x_max_uncropped - x_min_uncropped
    h_uncropped = y_max_uncropped - y_min_uncropped

    # Create transformation matrix that maps from ROI coordinates to output coordinates
    roi_to_output = np.array([
        [1, 0, -x_min_uncropped],
        [0, 1, -y_min_uncropped],
        [0, 0, 1]
    ])
    
    # Combine all transformations
    full_transform = roi_to_output @ final_transform
    
    # Apply the transformation to the ROI
    transformed_full = cv2.warpAffine(
        roi_original,
        full_transform[:2], # Take only first 2 rows for cv2.warpAffine
        (w_uncropped, h_uncropped),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # Now crop to image boundaries
    x_min = int(max(0, x_min_uncropped))
    y_min = int(max(0, y_min_uncropped))
    x_max = int(min(img_w, x_max_uncropped))
    y_max = int(min(img_h, y_max_uncropped))

    # Calculate offsets into uncropped image
    x_offset = int(x_min - x_min_uncropped)
    y_offset = int(y_min - y_min_uncropped)
    crop_w = x_max - x_min
    crop_h = y_max - y_min

    # Extract cropped region from full transformation
    transformed_cropped = transformed_full[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

    # Create output image and paste cropped result
    transformed_image = image.copy()
    transformed_image[y_min:y_max, x_min:x_max] = transformed_cropped

    # For debugging, save visualization of bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot original bbox
    original_corners = corners[:, :2]
    original_corners = np.vstack([original_corners, original_corners[0]])
    plt.plot(original_corners[:, 0], original_corners[:, 1], 'r-', linewidth=2, label='Original')

    # Plot uncropped transformed bbox
    uncropped_corners = np.array([
        [x_min_uncropped, y_min_uncropped],
        [x_max_uncropped, y_min_uncropped], 
        [x_max_uncropped, y_max_uncropped],
        [x_min_uncropped, y_max_uncropped],
        [x_min_uncropped, y_min_uncropped]
    ])
    plt.plot(uncropped_corners[:, 0], uncropped_corners[:, 1], 'g-', linewidth=2, label='Transformed')

    # Plot final cropped bbox
    cropped_corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max], 
        [x_min, y_max],
        [x_min, y_min]
    ])
    plt.plot(cropped_corners[:, 0], cropped_corners[:, 1], 'b-', linewidth=2, label='Cropped')

    plt.legend()
    plt.axis('off')
    plt.savefig('bbox_transformation.png')
    plt.close()

    return transformed_image

def inverse_warp_with_transformation_matrix_marco(A, roi_A, B, seg_map, transform_matrix):
    """
    Perform an inverse warping with affine transformation of a region from matrix A to B.
    
    Parameters:
    - A: Source tensor [51, 1, 4, 64, 64]
    - roi_A: Source region [x_min, x_max, y_min, y_max]
    - B: Target tensor [51, 1, 4, 64, 64] 
    - seg_map: Binary segmentation mask [64, 64]
    - transform_matrix: 3x3 affine transformation matrix
    """

    print("inverse_warp_with_transformation_matrix_marco")
    # x_min, x_max, y_min, y_max = roi_A
    # # Convert bbox format for apply_affine_transformation
    # bbox = [x_min/64.0, y_min/64.0, (x_max-x_min)/64.0, (y_max-y_min)/64.0]

   

    for i in range(B.shape[0]):
        # Process each channel separately
        for c in range(B.shape[2]):
            latent_2D = B[i, 0, c].cpu().numpy().astype(np.float32)  # Get single channel and ensure float32
            latent_2D, new_bbox = apply_affine_transform(latent_2D, roi_A, transform_matrix, is_mask=False)
            B[i, 0, c] = torch.from_numpy(latent_2D.astype(np.float32))  # Convert back to float

    # Process segmentation mask
    mask_2D = seg_map.astype(np.float32)  # Ensure float32
    mask_2D, _ = apply_affine_transform(mask_2D, roi_A, transform_matrix, is_mask=True)  # Apply same transform
    new_mask = torch.from_numpy(mask_2D > 0.5).to(A.device).bool()  # Convert back to tensor and ensure boolean with higher threshold

    # now apply the new_mask to the new_latents only inside the new_bbox
    M2 = np.ones((64, 64), dtype=np.float32)
    M2[new_bbox[2]:new_bbox[3], new_bbox[0]:new_bbox[1]] = new_mask[new_bbox[2]:new_bbox[3], new_bbox[0]:new_bbox[1]].float()
    B *= M2


    
    return B, new_mask

        

def inverse_warp_with_transformation_matrix(A, roi_A, B, seg_map, transform_matrix):
    """
    Perform an inverse warping with affine transformation of a region from matrix A to B.
    
    Parameters:
    - A: Source tensor [51, 1, 4, 64, 64]
    - roi_A: Source region [x_min, x_max, y_min, y_max]
    - B: Target tensor [51, 1, 4, 64, 64] 
    - seg_map: Binary segmentation mask [64, 64]
    - transform_matrix: 3x3 affine transformation matrix
    """



    # Prepare tensors
    A = A.squeeze(1)  # [51, 4, 64, 64]
    B = B.squeeze(1)
    
    # Extract source ROI dimensions
    x_min, x_max, y_min, y_max = roi_A
    h, w = y_max - y_min, x_max - x_min
    
    # Create coordinate grid for the source ROI
    y_range = torch.arange(h, dtype=torch.float32, device=A.device)
    x_range = torch.arange(w, dtype=torch.float32, device=A.device)
    y_coords, x_coords = torch.meshgrid(y_range, x_range)
    
    # Calculate center of image
    image_center_x = 32
    image_center_y = 32
    
    # Calculate ROI center
    roi_center_x = (x_min + x_max) / 2
    roi_center_y = (y_min + y_max) / 2
    
    # Translation to image center
    translation_to_center = np.array([
        [1, 0, image_center_x - roi_center_x],
        [0, 1, image_center_y - roi_center_y],
        [0, 0, 1]
    ])
    
    # Translation back from center
    translation_from_center = np.array([
        [1, 0, -(image_center_x - roi_center_x)],
        [0, 1, -(image_center_y - roi_center_y)],
        [0, 0, 1]
    ])
    
    # Combine transformations: translate to center -> rotate -> translate back
    final_transform = translation_from_center @ transform_matrix @ translation_to_center
    
    # Create homogeneous coordinates for the ROI
    x_coords = x_coords + x_min  # Offset to ROI position
    y_coords = y_coords + y_min  # Offset to ROI position
    ones = torch.ones_like(x_coords)
    source_coords = torch.stack([x_coords, y_coords, ones], dim=-1).float()
    
    # Transform coordinates using combined transformation
    transform = torch.from_numpy(final_transform).float().to(A.device)
    transformed_coords = torch.matmul(source_coords, transform.T)
    
    # Get target coordinates
    target_x = transformed_coords[..., 0].long().clamp(0, 63)
    target_y = transformed_coords[..., 1].long().clamp(0, 63)
    
    # Create mask from segmentation map and apply ROI
    seg_mask = torch.from_numpy(seg_map).float().to(A.device)
    roi_seg_mask = seg_mask[y_min:y_max, x_min:x_max]
    
    # Copy transformed pixels only where seg_map indicates
    for batch in range(A.shape[0]):
        for channel in range(A.shape[1]):
            for i in range(h):
                for j in range(w):
                    tx, ty = target_x[i, j], target_y[i, j]
                    if roi_seg_mask[i, j] > 0:  # Only transform pixels in seg_map
                        B[batch, channel, ty, tx] = A[batch, channel, y_min + i, x_min + j]
    
    # Restore dimensions
    A = A.unsqueeze(1)
    B = B.unsqueeze(1)
    
    # Create transformed mask by applying same transform to seg_map
    new_mask = torch.zeros((64, 64), dtype=bool, device=A.device)
    for i in range(h):
        for j in range(w):
            if roi_seg_mask[i, j] > 0:
                tx, ty = target_x[i, j], target_y[i, j]
                new_mask[ty, tx] = True
                
    return B, new_mask

def inverse_warp(A, roi_A, B, roi_B_target, seg_map):
    """
    Perform an inverse warping of a region of interest (ROI) from matrix A to matrix B.
    This function takes a specified region from the source matrix A, applies a mask to it,
    and then places the masked region into a specified location in the target matrix B.

    Parameters:
    - A: Source PyTorch tensor. This is the matrix from which the region of interest will be extracted.
    - roi_A: A list defining the region of interest in A, formatted as [x_min, x_max, y_min, y_max].
      These values specify the bounding box of the region to be extracted from A.
    - B: Target PyTorch tensor. This is the matrix where the extracted region will be placed.
    - roi_B_target: A list defining the target rectangle in B, formatted as [x_min_target, x_max_target, y_min_target, y_max_target].
      These values specify where the extracted region from A will be placed in B.
    - seg_map: A binary segmentation map as a NumPy array. This map is used to mask the region of interest in A,
      ensuring that only the relevant parts of the region are transferred to B.

    Returns:
    - B: The target matrix B with the region from A placed at the specified location.
    - new_mask: A boolean mask of the same size as B, indicating the new location of the region in B.
    """
    # Extract the coordinates of the region of interest from A
    x_min, x_max, y_min, y_max = roi_A
    # Extract the coordinates of the target region in B
    x_min_target, x_max_target, y_min_target, y_max_target = roi_B_target

    # Adjust the maximum x and y coordinates to ensure they do not exceed the boundaries of B
    x_max -= max(0, (x_max_target - 63))
    y_max -= max(0, (y_max_target - 63))

    # Remove the singleton dimension from A and B to simplify operations
    A = A.squeeze(1)
    B = B.squeeze(1)

    # Convert the segmentation map to a PyTorch tensor and expand its dimensions to match A
    seg_map = (
        torch.from_numpy(seg_map)
        .unsqueeze(0)  # Add a batch dimension
        .unsqueeze(0)  # Add a channel dimension
        .repeat(A.shape[0], A.shape[1], 1, 1)  # Repeat to match the dimensions of A
    ).float()

    # Extract the region of interest from A and apply the segmentation mask
    roi_content = (
        A[:, :, int(y_min):int(y_max), int(x_min):int(x_max)]
        * seg_map[:, :, int(y_min):int(y_max), int(x_min):int(x_max)]
    )

    # Extract the masked region from the segmentation map
    seg_roi_content = seg_map[:, :, int(y_min):int(y_max), int(x_min):int(x_max)]

    # Determine the shape of the region of interest
    roi_shape = roi_content.shape[-2:]

    # Place the masked region into the target matrix B at the specified location
    B[
        :,
        :,
        int(y_min_target):int(y_min_target) + roi_shape[0],
        int(x_min_target):int(x_min_target) + roi_shape[1],
    ] = roi_content

    # Restore the singleton dimension to A and B
    A = A.unsqueeze(1)
    B = B.unsqueeze(1)

    # Create a new boolean mask for B, indicating the location of the placed region
    new_mask = torch.zeros((64, 64), dtype=bool)
    seg_roi_shape = seg_roi_content.shape[-2:]
    new_mask[
        int(y_min_target):int(y_min_target) + seg_roi_shape[0],
        int(x_min_target):int(x_min_target) + seg_roi_shape[1],
    ] = seg_roi_content[0][0]

    return B, new_mask


def plot_feat(tensor_data, fname):
    import matplotlib.pyplot as plt

    # Convert the tensor to a NumPy array
    numpy_data = (
        tensor_data.squeeze().cpu().numpy()
    )  # Remove dimensions of size 1 and convert to NumPy array

    # Create a figure and a grid of subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Loop through the 4 images and display them in subplots
    for i in range(4):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(
            numpy_data[i], cmap="gray"
        )  # Assuming images are grayscale

    # Set titles for each subplot (optional)
    for i, ax in enumerate(axes.flat):
        ax.set_title(f"Feat {i + 1}")

    # Remove axis labels and ticks (optional)
    for ax in axes.flat:
        ax.axis("off")

    # Adjust spacing between subplots (optional)
    plt.tight_layout()

    # Show the plot
    plt.savefig(fname)
    plt.close()
    # plt.show()





@torch.no_grad()
def compose_latents_with_alignment(
    model_dict,
    latents_bg_lists,
    latents_all_list,
    mask_tensor_list,
    original_remove,
    change_objects,
    move_objects,
    num_inference_steps,
    overall_batch_size,
    height,
    width,
    align_with_overall_bboxes=True,
    overall_bboxes=None,
    horizontal_shift_only=False,
    bg_seed=1,
    **kwargs,
):
  
    print("MARCOOOOOOOO")
    if align_with_overall_bboxes and len(latents_all_list):
        expanded_overall_bboxes = utils.expand_overall_bboxes(overall_bboxes)
        latents_all_list, mask_tensor_list, offset_list = align_with_bboxes(
            latents_all_list,
            mask_tensor_list,
            bboxes=expanded_overall_bboxes,
            horizontal_shift_only=horizontal_shift_only,
        )
    else:
        offset_list = [(0.0, 0.0) for _ in range(len(latents_all_list))]

    # Compose Move Objects
    # import pdb

    # pdb.set_trace()
    # latents_all_list.append(latents_bg_lists)
    # mask_tensor_list.append(bg_mask)


    
    use_marco = True


    for obj_name, old_obj, new_obj, seg_map, all_latents in move_objects:

        # print(all_latents.shape)
        # exit()
        # x_min_old, x_max_old, y_min_old, y_max_old
        old_coords = coord_transform(old_obj, 64)  
        # x_min_new, x_max_new, y_min_new, y_max_new
        new_coords = coord_transform(new_obj, 64)
        new_latents = all_latents.clone()

       
        # define transformation
        if obj_name == "dog #1":
            transform_matrix = define_affine_transformation(rotation_angle = 0.0, translation = (0.0, 10.0), scaling = (0.7, 0.7))
        else:
            transform_matrix = define_affine_transformation(rotation_angle = 0.0, translation = (0.0, 0.0), scaling = (1.0, 1.0))


        if use_marco:
            # Call with correct parameter order
            new_latents, new_mask = inverse_warp_with_transformation_matrix_marco(
                A=all_latents,
                roi_A=old_coords,
                B=new_latents,
                seg_map=seg_map,
                transform_matrix=transform_matrix
            )
        else:
            new_latents, new_mask = inverse_warp(
                A=all_latents,
                roi_A=old_coords,
                B=new_latents,
                roi_B_target=new_coords,
                seg_map=seg_map
            )

        # Save the new mask as a PNG file
        mask_image = Image.fromarray((new_mask.cpu().numpy() * 255).astype(np.uint8))
        mask_image.save(f"new_mask_{obj_name.replace(' ', '_')}.png")

        # Save latents as images as PNG files
        temp_dir = "temp_latent_vis"
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f"Saving verifying intermediate results to {os.path.abspath(temp_dir)}...")
        step_size = max(len(new_latents) // 10, 1)
        for i in range(0, len(new_latents), step_size):
            latent = new_latents[i]
            latent_viz = latent / 0.18215
        
            # Visualize latent channels
            latent_channels = latent_viz[0, :3]
            latent_channels = (latent_channels - latent_channels.min()) / (
                latent_channels.max() - latent_channels.min()
            )
            tvt.ToPILImage()(latent_channels).save(
                os.path.join(temp_dir, f"latent_channels_{i:04d}.png")
            )

        print(obj_name)
        if obj_name == "dog #1":
            breakpoint()


        # plot_feat(new_latents[-1], "feat_after.png")
        # plot_feat(all_latents[-1], "feat_before.png")
        # plot_feat(new_latents[-1], "feat_after.png")
        # exit()
        # print(new_latents.shape)
        # print(latents_bg_lists.shape)
        # print(new_mask.shape)
        # import pdb

        # pdb.set_trace()
        # new_latents[
        #     :, :, :, y_min_new:y_max_new, x_min_new:x_max_new
        # ] = latents_bg_lists[:, :, :, y_min_old:y_max_old, x_min_old:x_max_old]
        # new_mask = torch.zeros((64, 64), dtype=bool)
        # new_mask[y_min_new:y_max_new, x_min_new:x_max_new] = True
        # plt.imsave((mew_latents[].cpu().numpy() * 255).astype(np.uint8)).save(
        #     "new_mask.png"
        # )
        # Image.fromarray((new_mask.cpu().numpy() * 255).astype(np.uint8)).save(
        #     "new_mask.png"
        # )
        # old_mask = torch.zeros((64, 64), dtype=bool)
        # old_mask[y_min_old:y_max_old, x_min_old:x_max_old] = True
        # Image.fromarray((old_mask.cpu().numpy() * 255).astype(np.uint8)).save(
        #     "old_mask.png"
        # )
        # exit()
        latents_all_list.append(new_latents)
        mask_tensor_list.append(new_mask)
        # np.save()
        # break
    # N = len(mask_tensor_list)
    # for i in range(N):
    #     np.save(f"object_latent_{i:02d}.npy", latents_all_list[i].cpu().numpy())
    #     np.save(f"object_mask_{i:02d}.npy", mask_tensor_list[i].cpu().numpy())
    # exit()
    # import pdb

    # pdb.set_trace()
    for mask, latents in change_objects:
        latents_all_list.append(latents)
        mask_tensor_list.append(torch.from_numpy(mask))

    fg_mask_union = torch.zeros((64, 64), dtype=bool)
    N = len(mask_tensor_list)
    for i in range(N):
        fg_mask_union |= mask_tensor_list[i]
    bg_mask = ~fg_mask_union
    bg_mask[original_remove == True] = False
    # Image.fromarray((bg_mask.cpu().numpy() * 255).astype(np.uint8)).save("bg_mask.png")
    # print("bg_mask.png")
    # exit()
    # img_src = np.array(Image.open("sdv2_generation/round_0/30.png"))
    # print("sdv2_generation/round_0/30.png")
    # still_need_remove = original_remove.clone()
    # still_need_remove[fg_mask_union == True] = False

    # exit()
    # bg_mask = ~still_need_remove
    # import numpy as np

    # Image.fromarray((bg_mask * 255).cpu().numpy().astype(np.uint8)).save("bg_mask.png")
    # print("bg_mask.png")
    # for i in range(N):
    #     print(i, flush=True)
    #     Image.fromarray(
    #         (mask_tensor_list[i] * 255).cpu().numpy().astype(np.uint8)
    #     ).save(f"fg{i}_mask.png")
    #     print(f"fg{i}_mask.png")
    # # plot_feat(latents_bg_lists[-1], "feat_bg.png")
    # for i in range(51):
    #     plot_feat(latents_bg_lists[i] * bg_mask, f"vis_feat/BG{i}.png")
    # for j in range(N):
    #     for i in range(51):
    #         plot_feat(
    #             latents_all_list[j][i] * mask_tensor_list[j], f"vis_feat/fg{j}_{i}.png"
    #         )

    # exit()
    # input("OWO")

    # overlap_region = torch.zeros((64, 64), dtype=bool)
    # for i in range(N):
    #     overlap_region |= mask_tensor_list[i]
    # overlap_region |= bg_mask

    latents_all_list.append(latents_bg_lists)
    mask_tensor_list.append(bg_mask)

    composed_latents, foreground_indices = compose_latents(
        model_dict,
        latents_all_list,
        mask_tensor_list,
        num_inference_steps,
        overall_batch_size,
        height,
        width,
        bg_seed,
        **kwargs,
    )
    # composed_latents = latents_all_list[0].cuda()
    # foreground_indices = mask_tensor_list[0].cuda()
    # print(composed_latents.shape)
    # print(foreground_indices.shape)
    # exit()
    # for i in range(51):
    #     plot_feat(composed_latents[i], f"vis_feat/feat_final{i}.png")
    # exit()
    return composed_latents, foreground_indices, offset_list


def get_init_bg(model_dict):
    from models import pipelines

    print("haha here am I!", flush=True)
    init_image = Image.open("check.png")
    generator = torch.cuda.manual_seed(6666)
    cln_latents = pipelines.encode(model_dict, init_image, generator)

    vae, tokenizer, text_encoder, unet, scheduler, dtype = (
        model_dict.vae,
        model_dict.tokenizer,
        model_dict.text_encoder,
        model_dict.unet,
        model_dict.scheduler,
        model_dict.dtype,
    )

    input_embeddings = models.encode_prompts(
        prompts=["A forest"],
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        negative_prompt="",
        one_uncond_input_only=False,
    )
    all_latents = models.pipelines.invert(
        model_dict,
        cln_latents,
        input_embeddings,
        num_inference_steps=50,
        guidance_scale=2.5,
    )
    print(all_latents.shape)
    # all_latents_cpu = all_latents.cpu().numpy()
    # gen_latents = all_latents[0].cuda()
    # gen_latents *= scheduler.init_noise_sigma
    return all_latents


def get_input_latents_list(
    model_dict,
    latents_bg,
    bg_seed,
    fg_seed_start,
    fg_blending_ratio,
    height,
    width,
    so_prompt_phrase_box_list=None,
    so_boxes=None,
    verbose=False,
):
    """
    Note: the returned input latents are scaled by `scheduler.init_noise_sigma`
    """
    unet, scheduler, dtype = model_dict.unet, model_dict.scheduler, model_dict.dtype

    generator_bg = torch.manual_seed(
        bg_seed
    )  # Seed generator to create the inital latent noise

    # latents_bg_lists = get_init_bg(model_dict)
    # latents_bg = latents_bg_lists[1].cuda()
    if latents_bg is None:
        latents_bg = get_unscaled_latents(
            batch_size=1,
            in_channels=unet.config.in_channels,
            height=height,
            width=width,
            generator=generator_bg,
            dtype=dtype,
        )

    input_latents_list = []

    if so_boxes is None:
        # For compatibility
        so_boxes = [item[-1] for item in so_prompt_phrase_box_list]

    # change this changes the foreground initial noise
    for idx, obj_box in enumerate(so_boxes):
        H, W = height // 8, width // 8
        fg_mask = utils.proportion_to_mask(obj_box, H, W)
        # plt.imsave("fg_mask.jpg", fg_mask.cpu().numpy())
        # exit()
        if verbose:
            plt.imshow(fg_mask.cpu().numpy())
            plt.show()

        fg_seed = fg_seed_start + idx
        if fg_seed == bg_seed:
            # We should have different seeds for foreground and background
            fg_seed += 12345

        generator_fg = torch.manual_seed(fg_seed)
        latents_fg = get_unscaled_latents(
            batch_size=1,
            in_channels=unet.config.in_channels,
            height=height,
            width=width,
            generator=generator_fg,
            dtype=dtype,
        )

        input_latents = blend_latents(
            latents_bg, latents_fg, fg_mask, fg_blending_ratio=fg_blending_ratio
        )

        input_latents = input_latents * scheduler.init_noise_sigma

        input_latents_list.append(input_latents)

    latents_bg = latents_bg * scheduler.init_noise_sigma

    return input_latents_list, latents_bg