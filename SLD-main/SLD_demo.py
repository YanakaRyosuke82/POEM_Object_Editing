import os
import json
import copy
import shutil
import random
import numpy as np
import argparse
import configparser
from PIL import Image
import cv2
import logging


import torch
import diffusers

# Libraries heavily borrowed from LMD
import models
from models import sam
from utils import parse, utils

# SLD specific imports
from sld.detector import OWLVITV2Detector
from sld.sdxl_refine import sdxl_refine
from sld.utils import get_all_latents, run_sam, run_sam_postprocess, resize_image
from sld.llm_template import spot_object_template, spot_difference_template, image_edit_template
from sld.llm_chat import get_key_objects, get_updated_layout


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sld_debug.log'),
        logging.StreamHandler()
    ]
)


def load_custom_masks(data_entry):
    """Load custom masks and transforms from data entry"""
    logging.info("Checking for custom transforms in data entry")
    if "custom_transforms" not in data_entry:
        logging.info("No custom transforms found")
        return None
        
    custom_masks = {}
    for obj_name, transform_data in data_entry["custom_transforms"].items():
        logging.info(f"Loading custom transform for object: {obj_name}")
        try:
            # Load source mask
            source_mask_path = transform_data["source_mask"]
            logging.info(f"Loading source mask from: {source_mask_path}")
            source_mask = np.load(source_mask_path)
            
            # Load transform matrix
            logging.info(f"Loading transform matrix for {obj_name}")
            transform_matrix = np.array(transform_data["transform_matrix"])
            logging.info(f"Transform matrix: \n{transform_matrix}")
            
            custom_masks[obj_name] = (source_mask, transform_matrix)
            logging.info(f"Successfully loaded custom transform for {obj_name}")
            logging.info(f"Mask shape: {source_mask.shape}, Matrix shape: {transform_matrix.shape}")
        except Exception as e:
            logging.error(f"Error loading custom transform for {obj_name}: {str(e)}")
            raise
    
    return custom_masks


# Operation #1: Addition (The code is in sld/image_generator.py)

# Operation #2: Deletion (Preprocessing region mask for removal)
def get_remove_region(entry, remove_objects, move_objects, preserve_objs, models, config):
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
def get_repos_info(entry, move_objects, models, config, custom_masks=None):
    """Updates a list of objects to be moved/reshaped using either SAM-generated masks or custom binary masks."""
    if not move_objects:
        logging.info("No objects to reposition")
        return []  # Return empty list instead of move_objects
        
    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape
    logging.info(f"Source image shape: {image_source.shape}")
    inv_seed = int(config.get("SLD", "inv_seed"))

    new_move_objects = []
    for item in move_objects:
        obj_name = item[0][0]  # Get object name
        logging.info(f"Processing object: {obj_name}")
        
        if False or (custom_masks is not None and obj_name in custom_masks):
            logging.info(f"Using custom mask for {obj_name}")
            try:
                # Use provided custom mask and transform
                mask, transform = custom_masks[obj_name]
                logging.info(f"Mask shape: {mask.shape}, Transform matrix shape: {transform.shape}")
                old_object_region = mask.astype(np.bool_)

                
                # # Save the old object region to disk for visualization
                # old_object_region_path = os.path.join(os.getcwd(), f"old_object_region_{obj_name}.png")
                # cv2.imwrite(old_object_region_path, old_object_region.astype(np.uint8) * 255)
                # logging.info(f"Saved old object region for {obj_name} to {old_object_region_path}")

                
                
                # Apply transformation to the image region
                logging.info("Applying perspective transform")
                transformed_img = cv2.warpPerspective(image_source, transform, (W, H))
                
                # Get latents for the transformed image
                logging.info("Generating latents for transformed image")
                all_latents, _ = get_all_latents(transformed_img, models, inv_seed)
                
                if all_latents is None:
                    logging.error(f"Failed to generate latents for {obj_name}")
                    raise
                    
                # For custom transforms, use None for the target bbox to ignore layout suggestions
                new_move_objects.append([
                    obj_name,  # name
                    None,      # object data (None for custom masks)
                    None,      # target bbox (None to ignore layout suggestions)
                    old_object_region,  # mask
                    all_latents  # latents
                ])
                logging.info(f"Successfully processed custom transform for {obj_name}")
            except Exception as e:
                logging.error(f"Error processing custom transform for {obj_name}: {str(e)}")
                raise
        else:
            logging.info(f"Using SAM-based logic for {obj_name}")
            try:
                new_img, obj = resize_image(image_source, item[0][1], item[1][1])
                logging.info(f"Resized image shape: {new_img.shape}")
                
                old_object_region = run_sam_postprocess(
                    run_sam(obj, new_img, models), H, W, config
                ).astype(np.bool_)
                logging.info(f"Generated mask shape: {old_object_region.shape}")
                
                all_latents, _ = get_all_latents(new_img, models, inv_seed)
                new_move_objects.append(
                    [item[0][0], obj, item[1][1], old_object_region, all_latents]
                )
                logging.info(f"Successfully processed SAM-based transform for {obj_name}")
            except Exception as e:
                logging.error(f"Error in SAM-based processing for {obj_name}: {str(e)}")
                raise

    logging.info(f"Returning {len(new_move_objects)} processed objects")
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
        # Run diffedit
        old_object_region = run_sam_postprocess(run_sam(obj[1], image_source, models), H, W, config)
        old_object_region = old_object_region.astype(np.bool_)[np.newaxis, ...]

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
            mask_image=old_object_region,
            image_latents=image_latents,
            guidance_scale=float(config.get("SLD", "diffedit_guidance_scale")),
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
            negative_prompt="",
        ).images[0]

        all_latents, _ = get_all_latents(np.array(image), models, inv_seed)
        new_change_objects.append(
            [
                old_object_region[0],
                all_latents,
            ]
        )
    return new_change_objects


def correction(
    entry, add_objects, move_objects,
    remove_region, change_attr_objects,
    models, config
):
    logging.info("Starting correction function")
    logging.info(f"Move objects: {len(move_objects)}")
    logging.info(f"Add objects: {len(add_objects)}")
    logging.info(f"Change objects: {len(change_attr_objects)}")
    
    spec = {
        "add_objects": add_objects,
        "move_objects": move_objects,
        "prompt": entry["instructions"],
        "remove_region": remove_region,
        "change_objects": change_attr_objects,
        "all_objects": entry["llm_suggestion"],
        "bg_prompt": entry["bg_prompt"],
        "extra_neg_prompt": entry["neg_prompt"],
    }
    
    logging.info("Created spec dictionary")
    image_source = np.array(Image.open(entry["output"][-1]))
    # Background latent preprocessing
    logging.info("Getting background latents")
    all_latents, _ = get_all_latents(image_source, models, int(config.get("SLD", "inv_seed")))
    


    logging.info("Inspecting move_objects in spec for correction function")
    if isinstance(spec["move_objects"], list):
        logging.info(f"Move objects are iterable")
    else:
        logging.error("Move objects are not iterable")
    logging.info("Running image generator")
    ret_dict = image_generator.run(
        spec,
        fg_seed_start=int(config.get("SLD", "fg_seed")),
        bg_seed=int(config.get("SLD", "bg_seed")),
        bg_all_latents=all_latents,
        frozen_step_ratio=float(config.get("SLD", "frozen_step_ratio")),
    )
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
        questions = (
            f"User Prompt: {prompt}\nCurrent Objects: {det_results}\nReasoning:\n"
        )
        if mode == "self_correction":
            message = spot_difference_template + questions
        else:
            message = image_edit_template + questions
        llm_suggestions = get_updated_layout(message, config)
        return llm_suggestions[0]
    else:
        return data["llm_layout_suggestions"]


if __name__ == "__main__":

    # clear contents of the log file
    with open('sld_debug.log', 'w') as f:
        f.truncate(0)

    logging.info("Starting SLD demo")
    # create argument parser
    parser = argparse.ArgumentParser(description="Demo for the SLD pipeline")
    parser.add_argument("--json-file", type=str, default="demo/self_correction/data.json", help="Path to the json file")
    parser.add_argument("--input-dir", type=str, default="demo/self_correction/src_image", help="Path to the input directory")
    parser.add_argument("--output-dir", type=str, default="demo/self_correction/results", help="Path to the output directory")
    parser.add_argument("--mode", type=str, default="self_correction", help="Mode of the demo", choices=["self_correction", "image_editing"])
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
    models.sd_key = "gligen/diffusers-generation-text-box"
    models.sd_version = "sdv1.4"
    diffusion_scheduler = None

    models.model_dict = models.load_sd(
        key=models.sd_key,
        use_fp16=False,
        load_inverse_scheduler=True,
        scheduler_cls=diffusers.schedulers.__dict__[diffusion_scheduler]
        if diffusion_scheduler is not None
        else None,
    )
    sam_model_dict = sam.load_sam()
    models.model_dict.update(sam_model_dict)
    from sld import image_generator

    det = OWLVITV2Detector()
    # Iterate through the json file
    for idx in range(len(data)):

        try:
            # Reset random seeds
            default_seed = int(config.get("SLD", "default_seed"))
            torch.manual_seed(default_seed)
            np.random.seed(default_seed)
            random.seed(default_seed)

            # Load the image and prompt
            rel_fname = data[idx]["input_fname"]
            fname = os.path.join(args.input_dir, f"{rel_fname}.png")
            
            prompt = data[idx]["prompt"]
            dirname = os.path.join(save_dir, data[idx]["output_dir"])
            os.makedirs(dirname, exist_ok=True)

            output_fname = os.path.join(dirname, f"initial_image.png")
            shutil.copy(fname, output_fname)

            print("-" * 5 + f" [Self-Correcting {fname}] " + "-" * 5)
            print(f"Target Textual Prompt: {prompt}")

            # Step 1: Spot Objects with LLM
            llm_parsed_prompt = spot_objects(prompt, data[idx], config)
            entry = {"instructions": prompt, "output": [fname], "generator": data[idx]["generator"],
                     "objects": llm_parsed_prompt["objects"], 
                     "bg_prompt": llm_parsed_prompt["bg_prompt"],
                     "neg_prompt": llm_parsed_prompt["neg_prompt"]
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

            attr_threshold = float(config.get(entry["generator"], "attr_detection_threshold", fallback=default_attr_threshold))
            prim_threshold = float(config.get(entry["generator"], "prim_detection_threshold", fallback=default_prim_threshold))
            nms_threshold = float(config.get(entry["generator"], "nms_threshold", fallback=default_nms_threshold))
            det_results = det.run(prompt, entry["objects"], entry["output"][-1],
                                  attr_detection_threshold=attr_threshold, 
                                  prim_detection_threshold=prim_threshold, 
                                  nms_threshold=nms_threshold)

            print("-" * 5 + f" Getting Modification Suggestions " + "-" * 5)

            # Step 3: Spot difference between detected results and initial prompts
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
            if (total_ops == 0):
                print("-" * 5 + f" Results " + "-" * 5)
                output_fname = os.path.join(dirname, f"final_{rel_fname}.png")
                shutil.copy(entry["output"][-1], output_fname)
                print("* No changes to apply!")
                print(f"* Output File: {output_fname}")
                # Shortcut to proceed to the next round!
                continue

            # Step 4: T2I Ops: Addition / Deletion / Repositioning / Attr. Modification
            print("-" * 5 + f" Image Manipulation " + "-" * 5)

            deletion_region = get_remove_region(
                entry, deletion_objs, repositioning_objs, [], models, config
            )
            
            # Load custom masks if available
            logging.info("Attempting to load custom masks")
            custom_masks = load_custom_masks(data[idx])
            if custom_masks:
                logging.info(f"Loaded custom masks for objects: {list(custom_masks.keys())}")
            
            logging.info("Processing repositioning objects")
            repositioning_objs = get_repos_info(
                entry, repositioning_objs, models, config, custom_masks=custom_masks
            )
            attr_modification_objs = get_attrmod_latent(
                entry, attr_modification_objs, models, config
            )
            
            if repositioning_objs is None or len(repositioning_objs) == 0:
                logging.warning("No objects to reposition after processing")
                repositioning_objs = []

            logging.info("Calling correction function")
            ret_dict = correction(
                entry, addition_objs, repositioning_objs,
                deletion_region, attr_modification_objs, 
                models, config
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
        except Exception as e:
            logging.error(f"Error processing entry {idx}: {str(e)}")
            raise