import os
import json
import copy
import shutil
import random
import numpy as np
import argparse
import configparser
from PIL import Image


import torch
import diffusers

# Libraries heavily borrowed from LMD
import models_marco
from models_marco import sam
from utils import parse, utils

# SLD specific imports
from sld.detector import OWLVITV2Detector
from sld.sdxl_refine import sdxl_refine
from sld.utils import get_all_latents, run_sam, run_sam_postprocess, resize_image
from sld.llm_template import spot_object_template, spot_difference_template, image_edit_template
from sld.llm_chat import get_key_objects, get_updated_layout


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Operation #1: Addition (The code is in sld/image_generator.py)


# Operation #2: Deletion (Preprocessing region mask for removal)
def get_remove_region(entry,  remove_objects, move_objects, preserve_objs, models, config):
    """Generate a region mask for removal given bounding box info."""

    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape

    # if no remove objects, set zero to the whole mask
    if (len(remove_objects) + len(move_objects)) == 0:
        remove_region = np.zeros((W // 8, H // 8), dtype=np.int64)
        return remove_region

    # Otherwise, run the SAM segmentation to locate target regions
    remove_items = remove_objects + [x[0] for x in move_objects]
    remove_mask = np.zeros((H, W, 3), dtype=bool)
    for obj in remove_items:
        masks = run_sam(bbox=obj[1], image_source=image_source, models=models)
        remove_mask = remove_mask | masks

    # Preserve the regions that should not be removed
    preserve_mask = np.zeros((H, W, 3), dtype=bool)
    for obj in preserve_objs:
        masks = run_sam(bbox=obj[1], image_source=image_source, models=models)
        preserve_mask = preserve_mask | masks
    # Process the SAM mask by averaging, thresholding, and dilating.
    preserve_region = run_sam_postprocess(preserve_mask, H, W, config)
    remove_region = run_sam_postprocess(remove_mask, H, W, config)
    remove_region = np.logical_and(remove_region, np.logical_not(preserve_region))
    return remove_region


# Operation #3: Repositioning (Preprocessing latent)
def get_repos_info(entry, move_objects, models, config):
    """
    Updates a list of objects to be moved / reshaped, including resizing images and generating masks.
    * Important: Perform image reshaping at the image-level rather than the latent-level.
    * Warning: For simplicity, the object is not positioned to the center of the new region...
    """

    # if no remove objects, set zero to the whole mask
    if not move_objects:
        return move_objects
    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))

    new_move_objects = []
    for item in move_objects:
        new_img, obj = resize_image(image_source, item[0][1], item[1][1])
        old_object_region = run_sam_postprocess(run_sam(obj, new_img, models), H, W, config).astype(np.bool_)
        all_latents, _ = get_all_latents(new_img, models, inv_seed)
        new_move_objects.append([item[0][0], obj, item[1][1], old_object_region, all_latents])

    return new_move_objects


# Operation #4: Attribute Modification (Preprocessing latent)
def get_attrmod_latent(entry, change_attr_objects, models, config):
    """
    Processes objects with changed attributes to generate new latents and the name of the modified objects.

    Parameters:
    entry (dict): A dictionary containing output data.
    change_attr_objects (list): A list of objects with changed attributes.
    models (Model): The models used for processing.
    inv_seed (int): Seed for inverse generation.

    Returns:
    list: A list containing new latents and names of the modified objects.
    """
    if len(change_attr_objects) == 0:
        return [] 
    from diffusers import StableDiffusionDiffEditPipeline
    from diffusers import DDIMScheduler, DDIMInverseScheduler

    img = Image.open(entry["output"][-1])
    image_source = np.array(img)
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))

    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    new_change_objects = []
    for obj in change_attr_objects:
        # Get the original object region using SAM
        old_object_region = run_sam_postprocess(run_sam(obj[1], image_source, models), H, W, config)
        
        # Load the binary mask and ensure it's boolean
        mask_image = np.array(entry["mask_image"])
        mask_image = (mask_image > 0).astype(np.bool_)
        mask_image = mask_image[np.newaxis, ...]

        new_object = obj[0].split(" #")[0]
        base_object = new_object.split(" ")[-1]
        mask_prompt = f"a {base_object}"
        new_prompt = f"a {new_object}"

        image_latents = pipe.invert(
            image=img,
            prompt=mask_prompt,
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
        ).latents
        image = pipe(
            prompt=new_prompt,
            mask_image=mask_image,  # Use the loaded binary mask directly
            image_latents=image_latents,
            guidance_scale=float(config.get("SLD", "diffedit_guidance_scale")),
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
            negative_prompt="",
        ).images[0]

        all_latents, _ = get_all_latents(np.array(image), models, inv_seed)
        new_change_objects.append(
            [
                old_object_region,  # Keep original region for reference
                all_latents,
            ]
        )
    return new_change_objects


###### HERE IS WHERE THE MAGIC HAPPENS ######
def correction(entry, add_objects, move_objects, remove_region, change_attr_objects, models, config):
    """Performs image correction based on specified modifications and generates a new image.

    Args:
        entry (dict): Dictionary containing image metadata including instructions, prompts and output paths
        add_objects (list): Objects to be added to the image
        move_objects (list): Objects to be moved within the image
        remove_region (list): Regions to be removed from the image
        change_attr_objects (list): Objects whose attributes need to be changed
        models (dict): Dictionary containing the required ML models
        config (ConfigParser): Configuration parameters for the correction process

    Returns:
        dict: Dictionary containing the generated image and related metadata

    The function performs the following steps:
    1. Processes input specifications and prompts
    2. Loads and preprocesses the source image
    3. Extracts latent representations
    4. Visualizes intermediate latent states (for debugging)
    5. Generates the corrected image using the image generator
    """
    spec = {
        "add_objects": add_objects,
        "move_objects": move_objects,
        "prompt": entry["instructions"],
        "remove_region": remove_region,
        "change_objects": change_attr_objects,
        "all_objects": entry["llm_suggestion"],
        "bg_prompt": entry["bg_prompt"],
        "extra_neg_prompt": entry["neg_prompt"],
        "mask_image": entry["mask_image"]
    }
    image_source = np.array(Image.open(entry["output"][-1]))
    # Background latent preprocessing
    all_latents, _ = get_all_latents(
        image_source, models, int(config.get("SLD", "inv_seed"))
    )  # torch.Size([51, 1, 4, 64, 64])

    ######### MARCO #########
    # Save 10 evenly spaced latents as images
    num_samples = 15
    total_latents = all_latents.shape[0]
    indices = np.linspace(0, total_latents - 1, num_samples, dtype=int)

    # Create directory for saving latent visualizations
    latents_dir = os.path.join(os.path.dirname(entry["output"][-1]), "latent_viz")
    os.makedirs(latents_dir, exist_ok=True)


    # Save raw latent tensors
    for i in indices:
        # Get single latent tensor
        latent = all_latents[i : i + 1]  # Shape: (1, 1, 4, 64, 64)
        
        print(f"Latent {i} shape:", latent.shape)
        print(f"Latent {i} min/max:", latent.min().item(), latent.max().item())
        
        # Convert to numpy and remove batch/channel dims
        latent_np = latent.squeeze().cpu().numpy()  # Shape: (4, 64, 64)
        print(f"Numpy array shape:", latent_np.shape)
        
        # Normalize and convert first 3 channels to RGB
        rgb_channels = []
        for c in range(3):
            channel = latent_np[c]
            if channel.max() == channel.min():
                normalized = np.zeros_like(channel)
            else:
                normalized = ((channel - channel.min()) * 255 / (channel.max() - channel.min()))
            rgb_channels.append(normalized.astype(np.uint8))
            
        rgb_image = np.stack(rgb_channels, axis=-1)  # Shape: (64, 64, 3)
        print(f"RGB image shape:", rgb_image.shape)
        print(f"RGB value range:", rgb_image.min(), rgb_image.max())
        
        latent_image = Image.fromarray(rgb_image)

        # Save image
        filename = os.path.join(latents_dir, f"latent_{i:03d}.png")
        latent_image.save(filename)
        print(f"Saved latent {i} visualization to {filename}")
        print("-" * 40)

    ################################
    breakpoint()

    ret_dict = image_generator.run(
        spec,
        fg_seed_start=int(config.get("SLD", "fg_seed")),
        bg_seed=int(config.get("SLD", "bg_seed")),
        bg_all_latents=all_latents,
        frozen_step_ratio=float(config.get("SLD", "frozen_step_ratio"))
    )

    breakpoint()
    # check ret_dict.image
    return ret_dict


def spot_objects(prompt, data, config):
    # If the object list is not available, run the LLM to spot objects
    if data.get("llm_parsed_prompt") is None:
        questions = f"User Prompt: {prompt}\nReasoning:\n"
        message = spot_object_template + questions
        results = get_key_objects(message, config)
        return results[0]  # Extracting the object list
    else:
        return data["llm_parsed_prompt"]


def spot_differences(prompt, det_results, data, config, mode="self_correction"):
    if data.get("llm_layout_suggestions") is None:
        questions = f"User Prompt: {prompt}\nCurrent Objects: {det_results}\nReasoning:\n"
        if mode == "self_correction":
            message = spot_difference_template + questions
        else:
            message = image_edit_template + questions
        llm_suggestions = get_updated_layout(message, config)
        return llm_suggestions[0]
    else:
        return data["llm_layout_suggestions"]


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(description="Demo for the SLD pipeline")
    parser.add_argument("--json-file", type=str, default="demo/self_correction/data.json", help="Path to the json file")
    parser.add_argument(
        "--input-dir", type=str, default="demo/self_correction/src_image", help="Path to the input directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="demo/self_correction/results", help="Path to the output directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="self_correction",
        help="Mode of the demo",
        choices=["self_correction", "image_editing"],
    )
    parser.add_argument("--config", type=str, default="demo_config.ini", help="Path to the config file")
    args = parser.parse_args()

    # Open the json file configured for self-correction (a list of filenames with prompts and other info...)
    # Create the output directory
    with open(args.json_file) as f:
        data = json.load(f)
    save_dir = args.output_dir
    parse.img_dir = os.path.join(save_dir, "tmp_imgs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(parse.img_dir, exist_ok=True)

    # Read config
    config = configparser.ConfigParser()
    config.read(args.config)

    # Load models
    models_marco.sd_key = "gligen/diffusers-generation-text-box"
    models_marco.sd_version = "sdv1.4"
    diffusion_scheduler = None

    models_marco.model_dict = models_marco.load_sd(
        key=models_marco.sd_key,
        use_fp16=False,
        load_inverse_scheduler=True,
        scheduler_cls=diffusers.schedulers.__dict__[diffusion_scheduler] if diffusion_scheduler is not None else None,
    )
    sam_model_dict = sam.load_sam()
    models_marco.model_dict.update(sam_model_dict)
    from sld import image_generator

    det = OWLVITV2Detector()
    # Iterate through the json file
    for idx in range(len(data)):

        # Reset random seeds
        default_seed = int(config.get("SLD", "default_seed"))
        torch.manual_seed(default_seed)
        np.random.seed(default_seed)
        random.seed(default_seed)

        # Load the image and prompt
        rel_fname = data[idx]["input_fname"]
        fname = os.path.join(args.input_dir, f"{rel_fname}.png")  # demo/self_correction/src_image_marco/elephant.png'

        prompt = data[idx]["prompt"]
        dirname = os.path.join(save_dir, data[idx]["output_dir"])
        os.makedirs(dirname, exist_ok=True)

        output_fname = os.path.join(dirname, f"initial_image.png")
        shutil.copy(fname, output_fname)

        print("-" * 5 + f" [Self-Correcting {fname}] " + "-" * 5)
        print(f"Target Textual Prompt: {prompt}")

        # Step 1: Spot Objects with LLM
        llm_parsed_prompt = spot_objects(prompt, data[idx], config)

        mask_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/src/SLD_main_bak/demo/image_editing/src_image_marco/elephant_transformed_mask.png"
        mask_image = Image.open(mask_path)
        entry = {
            "instructions": prompt,
            "output": [fname],
            "generator": data[idx]["generator"],
            "objects": llm_parsed_prompt["objects"],
            "bg_prompt": llm_parsed_prompt["bg_prompt"],
            "neg_prompt": llm_parsed_prompt["neg_prompt"],
            "mask_image" : mask_image
        }
        print("-" * 5 + f" Parsing Prompts " + "-" * 5)
        print(f"* Objects: {entry['objects']}")
        print(f"* Background: {entry['bg_prompt']}")
        print(f"* Negation: {entry['neg_prompt']}")

        # Step 2: Run open vocabulary detector
        print("-" * 5 + f" Running Detector " + "-" * 5)
        default_attr_threshold = float(config.get("SLD", "attr_detection_threshold"))
        default_prim_threshold = float(config.get("SLD", "prim_detection_threshold"))
        default_nms_threshold = float(config.get("SLD", "nms_threshold"))

        attr_threshold = float(
            config.get(entry["generator"], "attr_detection_threshold", fallback=default_attr_threshold)
        )
        prim_threshold = float(
            config.get(entry["generator"], "prim_detection_threshold", fallback=default_prim_threshold)
        )
        nms_threshold = float(config.get(entry["generator"], "nms_threshold", fallback=default_nms_threshold))
        det_results = det.run(
            prompt,
            entry["objects"],
            entry["output"][-1],
            attr_detection_threshold=attr_threshold,
            prim_detection_threshold=prim_threshold,
            nms_threshold=nms_threshold,
        )

        # Step 3: Spot difference between detected results and initial prompts
        print("-" * 5 + f" Getting Modification Suggestions " + "-" * 5)
        llm_suggestions = spot_differences(prompt, det_results, data[idx], config, mode=args.mode)

        print(f"* Detection Restuls: {det_results}")
        print(f"* LLM Suggestions: {llm_suggestions}")
        entry["det_results"] = copy.deepcopy(det_results)
        entry["llm_suggestion"] = copy.deepcopy(llm_suggestions)
        # Compare the two layouts to know where to update
        (
            preserve_objs,
            deletion_objs,
            addition_objs,
            repositioning_objs,
            attr_modification_objs,
        ) = det.parse_list(det_results, llm_suggestions)

        print("-" * 5 + f" Editing Operations " + "-" * 5)
        print(f"* Preservation: {preserve_objs}")
        print(f"* Addition: {addition_objs}")
        print(f"* Deletion: {deletion_objs}")
        print(f"* Repositioning: {repositioning_objs}")
        print(f"* Attribute Modification: {attr_modification_objs}")
        total_ops = len(deletion_objs) + len(addition_objs) + len(repositioning_objs) + len(attr_modification_objs)
        # Visualization
        parse.show_boxes(
            gen_boxes=entry["det_results"],
            additional_boxes=entry["llm_suggestion"],
            img=np.array(Image.open(entry["output"][-1])).astype(np.uint8),
            fname=os.path.join(dirname, "det_result_obj.png"),
        )
        # Check if there are any changes to apply
        if total_ops == 0:
            print("-" * 5 + f" Results " + "-" * 5)
            output_fname = os.path.join(dirname, f"final_{rel_fname}.png")
            shutil.copy(entry["output"][-1], output_fname)
            print("* No changes to apply!")
            print(f"* Output File: {output_fname}")
            # Shortcut to proceed to the next round!
            continue

        # Step 4: T2I Ops: Addition / Deletion / Repositioning / Attr. Modification
        ############################################################ DRAWING REGION MASK #  ###########################################################

        print("-" * 5 + f" Image Manipulation " + "-" * 5)

        deletion_region = get_remove_region(entry, deletion_objs, repositioning_objs, [], models_marco, config)
        repositioning_objs = get_repos_info(entry, repositioning_objs, models_marco, config)
        attr_modification_objs = get_attrmod_latent(entry, attr_modification_objs, models_marco, config)

        ##### MAIN MAGIC #####
        ret_dict = correction(
            entry, addition_objs, repositioning_objs, deletion_region, attr_modification_objs, models_marco, config
        )
        # Save an intermediate file without the SDXL refinement
        curr_output_fname = os.path.join(dirname, f"intermediate_{rel_fname}.png")
        Image.fromarray(ret_dict.image).save(curr_output_fname)
        print("-" * 5 + f" Results " + "-" * 5)
        print("* Output File (Before SDXL): ", curr_output_fname)
        utils.free_memory()

        # Can run this if applying SDXL as the refine process
        sdxl_output_fname = os.path.join(dirname, f"final_{rel_fname}.png")
        if args.mode == "self_correction":
            sdxl_refine(prompt, curr_output_fname, sdxl_output_fname)
        else:
            # For image editing, the prompt should be updated
            sdxl_refine(ret_dict.final_prompt, curr_output_fname, sdxl_output_fname)
        print("* Output File (After SDXL): ", sdxl_output_fname)
