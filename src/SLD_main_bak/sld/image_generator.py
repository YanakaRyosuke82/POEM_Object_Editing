import torch
import models
import utils
import numpy as np
from easydict import EasyDict

from models import pipelines, sam, model_dict
from utils import parse, guidance, attn, latents, vis
from utils.latents import get_scaled_latents
from sld.utils import DEFAULT_SO_NEGATIVE_PROMPT, DEFAULT_OVERALL_NEGATIVE_PROMPT

# import inflect
# p = inflect.engine()

vae, tokenizer, text_encoder, unet, scheduler, dtype = (
    model_dict.vae,
    model_dict.tokenizer,
    model_dict.text_encoder,
    model_dict.unet,
    model_dict.scheduler,
    model_dict.dtype,
)

version = "sld"

# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8  # size of the latent
guidance_scale = 2.5  # Scale for classifier-free guidance

# batch size: set to 1
overall_batch_size = 1

# attn keys for semantic guidance
guidance_attn_keys = pipelines.DEFAULT_GUIDANCE_ATTN_KEYS

# discourage masks with confidence below
discourage_mask_below_confidence = 0.85

# discourage masks with iou (with coarse binarized attention mask) below
discourage_mask_below_coarse_iou = 0.25

offload_cross_attn_to_cpu = False


def generate_single_object_with_box(
    prompt,
    box,
    phrase,
    word,
    input_latents,
    input_embeddings,
    semantic_guidance_kwargs,
    obj_attn_key,
    saved_cross_attn_keys,
    sam_refine_kwargs,
    num_inference_steps,
    gligen_scheduled_sampling_beta=0.3,
    verbose=False,
    visualize=True,
    **kwargs,
):
    bboxes, phrases, words = [box], [phrase], [word]

    if verbose:
        print(f"Getting token map (prompt: {prompt})")
    object_positions, word_token_indices = guidance.get_phrase_indices(
        tokenizer=tokenizer,
        prompt=prompt,
        phrases=phrases,
        words=words,
        return_word_token_indices=True,
        # Since the prompt for single object is from background prompt + object name, we will not have the case of not found
        add_suffix_if_not_found=False,
        verbose=verbose,
    )

    # phrases only has one item, so we select the first item in word_token_indices
    word_token_index = word_token_indices[0]

    if verbose:
        print("word_token_index:", word_token_index)
    (
        latents,
        single_object_images,
        saved_attns,
        single_object_pil_images_box_ann,
        latents_all,
    ) = pipelines.generate_gligen(
        model_dict,
        input_latents,
        input_embeddings,
        num_inference_steps,
        bboxes,
        phrases,
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
        guidance_scale=7.5,
        return_saved_cross_attn=True,
        semantic_guidance=True,
        semantic_guidance_bboxes=bboxes,
        semantic_guidance_object_positions=object_positions,
        semantic_guidance_kwargs=semantic_guidance_kwargs,
        saved_cross_attn_keys=[obj_attn_key, *saved_cross_attn_keys],
        return_cond_ca_only=True,
        return_token_ca_only=word_token_index,
        offload_cross_attn_to_cpu=offload_cross_attn_to_cpu,
        return_box_vis=True,
        save_all_latents=True,
        dynamic_num_inference_steps=True,
        **kwargs,
    )
    # `saved_cross_attn_keys` kwargs may have duplicates

    utils.free_memory()

    single_object_pil_image_box_ann = single_object_pil_images_box_ann[0]

    if visualize:
        print("Single object image")
        vis.display(single_object_pil_image_box_ann)
    mask_selected, conf_score_selected = sam.sam_refine_box(
        sam_input_image=single_object_images[0],
        box=box,
        model_dict=model_dict,
        verbose=verbose,
        **sam_refine_kwargs,
    )

    mask_selected_tensor = torch.tensor(mask_selected)


    return (
        latents_all,
        mask_selected_tensor,
        saved_attns,
        single_object_pil_image_box_ann,
    )


def get_masked_latents_all_list(
    so_prompt_phrase_word_box_list,
    input_latents_list,
    so_input_embeddings,
    verbose=False,
    **kwargs,
):
    latents_all_list, mask_tensor_list, saved_attns_list, so_img_list = [], [], [], []

    if not so_prompt_phrase_word_box_list:
        return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list

    so_uncond_embeddings, so_cond_embeddings = so_input_embeddings

    for idx, ((prompt, phrase, word, box), input_latents) in enumerate(
        zip(so_prompt_phrase_word_box_list, input_latents_list)
    ):
        so_current_cond_embeddings = so_cond_embeddings[idx : idx + 1]
        so_current_text_embeddings = torch.cat(
            [so_uncond_embeddings, so_current_cond_embeddings], dim=0
        )
        so_current_input_embeddings = (
            so_current_text_embeddings,
            so_uncond_embeddings,
            so_current_cond_embeddings,
        )

        latents_all, mask_tensor, saved_attns, so_img = generate_single_object_with_box(
            prompt,
            box,
            phrase,
            word,
            input_latents,
            input_embeddings=so_current_input_embeddings,
            verbose=verbose,
            **kwargs,
        )
        latents_all_list.append(latents_all)
        mask_tensor_list.append(mask_tensor)
        saved_attns_list.append(saved_attns)
        so_img_list.append(so_img)

    return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list


def convert_box(box):
    # box: x, y, w, h (in 512 format) -> x_min, y_min, x_max, y_max
    x_min, y_min = box[0], box[1]
    w_box, h_box = box[2], box[3]

    x_max, y_max = x_min + w_box, y_min + h_box

    return x_min, y_min, x_max, y_max


def convert_spec(spec):
    add_objects = spec["add_objects"]
    overall_prompt = spec["prompt"]
    so_prompt_phrase_word_box_list = []
    overall_phrases_words_boxes = []
    for obj in add_objects:
        raw_phrase = obj[0]
        word = obj[0].split("#")[0].strip()
        so_entry = (overall_prompt, f"a {word}", word, tuple(convert_box(obj[1])))
        phrase_entry = (f"a {word}", word, [tuple(convert_box(obj[1]))])
        so_prompt_phrase_word_box_list.append(so_entry)
        overall_phrases_words_boxes.append(phrase_entry)
    return so_prompt_phrase_word_box_list, overall_prompt, overall_phrases_words_boxes


def run(
    spec,
    bg_all_latents=None,
    bg_seed=1,
    overall_prompt_override="",
    fg_seed_start=20,
    frozen_step_ratio=0.5,
    num_inference_steps=50,
    loss_scale=5,
    loss_threshold=5.0,
    max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    max_index_step=0,
    overall_loss_scale=5,
    overall_loss_threshold=5.0,
    overall_max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    overall_max_index_step=30,
    so_gligen_scheduled_sampling_beta=0.4,
    overall_gligen_scheduled_sampling_beta=0.4,
    overall_fg_top_p=0.2,
    overall_bg_top_p=0.2,
    overall_fg_weight=1.0,
    overall_bg_weight=4.0,
    ref_ca_loss_weight=2.0,
    so_center_box=False,
    fg_blending_ratio=0.1,
    so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT,
    overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
    so_horizontal_center_only=True,
    align_with_overall_bboxes=False,
    horizontal_shift_only=True,
    use_fast_schedule=False,
    # Transfer the cross-attention from single object generation (with ref_ca_saved_attns)
    # Use reference cross attention to guide the cross attention in the overall generation
    use_ref_ca=True,
    use_autocast=True,
    verbose=False,
):
    """Main generation function that handles both adding new objects and removing existing ones.

    Args:
        spec: Generation specification containing prompts, objects to add/move/change, etc.
        bg_all_latents: Pre-computed background latents. If None, generates from scratch.
        bg_seed: Random seed for background generation
        overall_prompt_override: Optional override for the overall scene prompt
        fg_seed_start: Starting seed for foreground object generation (increments per object)
        frozen_step_ratio: Ratio of inference steps to keep frozen from input latents
        num_inference_steps: Total number of denoising steps
        loss_scale: Scale factor for per-object semantic loss
        loss_threshold: Minimum semantic loss threshold to apply optimization
        max_iter: Max optimization iterations per step for per-object generation
        max_index_step: Last step to apply per-object semantic guidance
        overall_loss_scale: Scale factor for overall scene semantic loss
        overall_loss_threshold: Minimum overall semantic loss threshold
        overall_max_iter: Max optimization iterations per step for overall scene
        overall_max_index_step: Last step to apply overall semantic guidance
        so_gligen_scheduled_sampling_beta: GLIGEN guidance strength for single objects
        overall_gligen_scheduled_sampling_beta: GLIGEN guidance strength for overall scene
        overall_fg_top_p: Top-p sampling threshold for foreground attention
        overall_bg_top_p: Top-p sampling threshold for background attention
        overall_fg_weight: Weight for foreground semantic guidance
        overall_bg_weight: Weight for background semantic guidance
        ref_ca_loss_weight: Weight for cross-attention transfer loss
        so_center_box: Whether to center object boxes in single object generation
        fg_blending_ratio: Ratio for blending foreground noise with background
        so_negative_prompt: Negative prompt for single object generation
        overall_negative_prompt: Negative prompt for overall scene generation
        so_horizontal_center_only: Only center objects horizontally
        align_with_overall_bboxes: Align masks/attention with overall scene boxes
        horizontal_shift_only: Only allow horizontal alignment shifts
        use_fast_schedule: Use faster schedule after transfer steps
        use_ref_ca: Use reference cross-attention for guidance
        use_autocast: Enable automatic mixed precision
        verbose: Print debug information

    Returns:
        EasyDict containing:
            image: Final generated image
            so_img_list: List of intermediate single object images
            final_prompt: Final prompt used for generation
    """
    #####_____________________________________________  Parse Arguments _____________________________________________
      

    frozen_step_ratio = min(max(frozen_step_ratio, 0.0), 1.0)
    frozen_steps = int(num_inference_steps * frozen_step_ratio)
    

    original_remove = spec["remove_region"].astype(np.bool_) # (64, 64)
    # print(
    #     "Key generation settings:",
    #     spec,
    #     bg_seed,
    #     fg_seed_start,
    #     frozen_step_ratio,
    #     so_gligen_scheduled_sampling_beta,
    #     overall_gligen_scheduled_sampling_beta,
    #     overall_max_index_step,
    # )

    if (
        len(spec["add_objects"])
        + len(spec["move_objects"])
        + len(spec["change_objects"])
    ) > 0:
        # Handle object addition/modification case
        add_spec = {}
        add_spec["prompt"] = spec["prompt"]
        add_spec["bg_prompt"] = spec["bg_prompt"]
        add_spec["extra_neg_prompt"] = spec["extra_neg_prompt"]
        add_spec["gen_boxes"] = []

        #### Add objects
        for obj in spec["add_objects"]:
            coord = [int(x * 512) for x in obj[1]]
            obj_name = f"a {obj[0].split('#')[0].strip()}"
            add_spec["gen_boxes"].append((obj_name, coord))
        (
            so_prompt_phrase_word_box_list,
            overall_prompt,
            overall_phrases_words_bboxes,
        ) = parse.convert_spec(add_spec, height, width, verbose=verbose)

        if overall_prompt_override and overall_prompt_override.strip():
            overall_prompt = overall_prompt_override.strip()

        overall_phrases, overall_words, overall_bboxes = (
            [item[0] for item in overall_phrases_words_bboxes],
            [item[1] for item in overall_phrases_words_bboxes],
            [item[2] for item in overall_phrases_words_bboxes],
        )

        # Center single object boxes if requested while keeping overall boxes at target positions
        if so_center_box:
            so_prompt_phrase_word_box_list = [
                (
                    prompt,
                    phrase,
                    word,
                    utils.get_centered_box(
                        bbox, horizontal_center_only=so_horizontal_center_only
                    ),
                )
                for prompt, phrase, word, bbox in so_prompt_phrase_word_box_list
            ]
            if verbose:
                print(
                    f"centered so_prompt_phrase_word_box_list: {so_prompt_phrase_word_box_list}"
                )
        so_boxes = [item[-1] for item in so_prompt_phrase_word_box_list]  # empty list

        # Append any extra negative prompts
        if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
            so_negative_prompt = spec["extra_neg_prompt"] + ", " + so_negative_prompt
            overall_negative_prompt = (
                spec["extra_neg_prompt"] + ", " + overall_negative_prompt
            )

        semantic_guidance_kwargs = dict(
            loss_scale=loss_scale,
            loss_threshold=loss_threshold,
            max_iter=max_iter,
            max_index_step=max_index_step,
            use_ratio_based_loss=False,
            guidance_attn_keys=guidance_attn_keys,
            verbose=False,
        )

        sam_refine_kwargs = dict(
            discourage_mask_below_confidence=discourage_mask_below_confidence,
            discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
            height=height,
            width=width,
            H=H,
            W=W,
        )

        #####_____________________________________________  Main generation pipeline using autocast for efficiency _____________________________________________
        with torch.autocast("cuda", enabled=use_autocast):
            so_prompts = [item[0] for item in so_prompt_phrase_word_box_list]
            if so_prompts:
                so_input_embeddings = models.encode_prompts(
                    prompts=so_prompts,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    negative_prompt=so_negative_prompt,
                    one_uncond_input_only=True,
                )
            else:
                so_input_embeddings = []

            # Initialize random noise for background
            generator = torch.manual_seed(
                bg_seed + 87
            )  

            initial_L = get_scaled_latents(
                51, unet.config.in_channels, 512, 512, generator, dtype, scheduler
            ).unsqueeze(1)


            # Convert original_remove numpy array to PyTorch tensor and move to CUDA device
            remove_region = torch.from_numpy(original_remove).float().cuda()

            # Blend background latents with initial noise in removal regions
            bg_all_latents = (
                bg_all_latents.cuda() * (1.0 - remove_region)
                + (initial_L.cuda()) * remove_region
            )
            initial_bg = bg_all_latents[0]
            
            # Generate input latents for each new object
            input_latents_list, latents_bg = latents.get_input_latents_list(
                model_dict,
                bg_seed=bg_seed,
                latents_bg=initial_bg,
                fg_seed_start=fg_seed_start,
                so_boxes=so_boxes,
                fg_blending_ratio=fg_blending_ratio,
                height=height,
                width=width,
                verbose=False,
            )

            # Determine when to switch to fast schedule if enabled
            if use_fast_schedule:
                fast_after_steps = (
                    max(frozen_steps, overall_max_index_step)
                    if use_ref_ca
                    else frozen_steps
                )
            else:
                fast_after_steps = None

            # Generate individual object latents and masks if needed for guidance
            if use_ref_ca or frozen_steps > 0:
                (
                    latents_all_list,
                    mask_tensor_list,
                    saved_attns_list,
                    so_img_list,
                ) = get_masked_latents_all_list(
                    so_prompt_phrase_word_box_list,
                    input_latents_list,
                    gligen_scheduled_sampling_beta=so_gligen_scheduled_sampling_beta,
                    semantic_guidance_kwargs=semantic_guidance_kwargs,
                    obj_attn_key=("down", 2, 1, 0),
                    saved_cross_attn_keys=guidance_attn_keys if use_ref_ca else [],
                    sam_refine_kwargs=sam_refine_kwargs,
                    so_input_embeddings=so_input_embeddings,
                    num_inference_steps=num_inference_steps,
                    fast_after_steps=fast_after_steps,
                    fast_rate=2,
                    verbose=verbose,
                )
            else:
                # Skip per-object generation if no guidance needed
                (latents_all_list, mask_tensor_list, saved_attns_list, so_img_list) = (
                    [],
                    [],
                    [],
                    [],
                )

            # Compose all latents with proper alignment
            (
                composed_latents,
                foreground_indices,
                offset_list,
            ) = latents.compose_latents_with_alignment(
                model_dict,
                bg_all_latents.cpu(),
                latents_all_list,
                mask_tensor_list,
                torch.from_numpy(original_remove),
                spec["change_objects"],
                spec["move_objects"],
                num_inference_steps,
                overall_batch_size,
                height,
                width,
                latents_bg=latents_bg,
                align_with_overall_bboxes=align_with_overall_bboxes,
                overall_bboxes=overall_bboxes,
                horizontal_shift_only=horizontal_shift_only,
                use_fast_schedule=use_fast_schedule,
                fast_after_steps=fast_after_steps,
                bg_seed=bg_seed,
            )

            # Process reference cross-attention if enabled
            if use_ref_ca:
                # ref_ca_saved_attns has the same hierarchy as bboxes
                ref_ca_saved_attns = []

                flattened_box_idx = 0
                for bboxes in overall_bboxes:
                    # bboxes: correspond to a phrase
                    ref_ca_current_phrase_saved_attns = []
                    for bbox in bboxes:
                        # each individual bbox
                        saved_attns = saved_attns_list[flattened_box_idx]
                        if align_with_overall_bboxes:
                            offset = offset_list[flattened_box_idx]
                            saved_attns = attn.shift_saved_attns(
                                saved_attns,
                                offset,
                                guidance_attn_keys=guidance_attn_keys,
                                horizontal_shift_only=horizontal_shift_only,
                            )
                        ref_ca_current_phrase_saved_attns.append(saved_attns)
                        flattened_box_idx += 1
                    ref_ca_saved_attns.append(ref_ca_current_phrase_saved_attns)

            # Prepare flattened object lists for final generation
            overall_bboxes_flattened, overall_phrases_flattened = [], []

            for obj_name, coords in spec["all_objects"]:
                obj_name = f"a {obj_name.split('#')[0].strip()}"
                x_min, y_min, h, w = coords
                new_coords = (x_min, y_min, x_min + h, y_min + w)
                overall_bboxes_flattened.append(new_coords)
                overall_phrases_flattened.append(obj_name)
                
            # Construct final scene prompt
            bg_prompt = spec["bg_prompt"]
            object_str = ", ".join(overall_phrases_flattened)
            overall_prompt = f"{bg_prompt} with {object_str}"

            # Get token indices for semantic guidance
            (
                overall_object_positions,
                overall_word_token_indices,
                overall_prompt,
            ) = guidance.get_phrase_indices(
                tokenizer=tokenizer,
                prompt=overall_prompt,
                phrases=overall_phrases,
                words=overall_words,
                verbose=verbose,
                return_word_token_indices=True,
                add_suffix_if_not_found=True,
            )
            
            # Encode final prompt embeddings
            overall_input_embeddings = models.encode_prompts(
                prompts=[overall_prompt],
                tokenizer=tokenizer,
                negative_prompt=overall_negative_prompt,
                text_encoder=text_encoder,
            )

            # Configure semantic guidance for final generation
            overall_semantic_guidance_kwargs = dict(
                loss_scale=overall_loss_scale,
                loss_threshold=overall_loss_threshold,
                max_iter=overall_max_iter,
                max_index_step=overall_max_index_step,
                fg_top_p=overall_fg_top_p,
                bg_top_p=overall_bg_top_p,
                fg_weight=overall_fg_weight,
                bg_weight=overall_bg_weight,
                ref_ca_word_token_only=True,
                ref_ca_last_token_only=True,
                ref_ca_saved_attns=ref_ca_saved_attns if use_ref_ca else None,
                word_token_indices=overall_word_token_indices,
                guidance_attn_keys=guidance_attn_keys,
                ref_ca_loss_weight=ref_ca_loss_weight,
                use_ratio_based_loss=False,
                verbose=verbose,
            )

            # Generate with composed latents

            # Foreground should be frozen
            frozen_mask = (foreground_indices != 0).to(torch.float32).cuda()
            
            # TODO: It seems like there are some bugs
            print(f"Final prompt: {overall_prompt}")
            
            # Final generation with all guidance
            with torch.autocast("cuda", enabled=use_autocast):
                _, images, _ = pipelines.generate_gligen_final(
                    model_dict,
                    composed_latents,
                    overall_input_embeddings,
                    num_inference_steps,
                    overall_bboxes_flattened,
                    overall_phrases_flattened,
                    guidance_scale=guidance_scale,
                    gligen_scheduled_sampling_beta=overall_gligen_scheduled_sampling_beta,
                    semantic_guidance=True,
                    semantic_guidance_bboxes=overall_bboxes,
                    semantic_guidance_object_positions=overall_object_positions,
                    semantic_guidance_kwargs=overall_semantic_guidance_kwargs,
                    frozen_steps=frozen_steps,
                    frozen_mask=frozen_mask,
                    initial_bg=None,
                    return_saved_cross_attn=True,
                )

            return EasyDict(image=images[0], so_img_list=so_img_list, final_prompt=overall_prompt)
    else:
        # Handle removal-only case
        frozen_mask = torch.from_numpy(~original_remove).to(torch.float32).cuda()
        # composed_latents = bg_all_latents[1:]
        # prompt = "A realistic cartoon-style painting"
        prompt = spec["bg_prompt"]
        # prompt = spec["prompt"]
        if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
            overall_negative_prompt = (
                spec["extra_neg_prompt"] + ", " + DEFAULT_OVERALL_NEGATIVE_PROMPT
            )
        else:
            overall_negative_prompt = DEFAULT_OVERALL_NEGATIVE_PROMPT
            
        # Encode prompt embeddings
        overall_input_embeddings = models.encode_prompts(
            prompts=[prompt],
            tokenizer=models.model_dict.tokenizer,
            text_encoder=models.model_dict.text_encoder,
            negative_prompt=overall_negative_prompt,
            one_uncond_input_only=False,
        )
        composed_latents = bg_all_latents.cuda()

        # Initialize random noise for removed regions
        generator = torch.manual_seed(
            bg_seed
        )
        
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
        
        # Replace removed regions with random noise
        frozen_mask_expanded = frozen_mask[None, None, None, ...].repeat(51, 1, 4, 1, 1)
        composed_latents[frozen_mask_expanded == 0] = latents_bg[
            frozen_mask_expanded == 0
        ]

        so_img_list = None

        # No object guidance needed for removal
        overall_bboxes_flattened = []
        overall_phrases_flattened = []
        overall_bboxes = []
        overall_object_positions = []
        overall_semantic_guidance_kwargs = None

        # Generate final image
        with torch.autocast("cuda", enabled=use_autocast):
            _, images = pipelines.generate_gligen_final(
                model_dict,
                composed_latents,
                overall_input_embeddings,
                num_inference_steps,
                overall_bboxes_flattened,
                overall_phrases_flattened,
                guidance_scale=guidance_scale,
                gligen_scheduled_sampling_beta=overall_gligen_scheduled_sampling_beta,
                semantic_guidance=True,
                semantic_guidance_bboxes=overall_bboxes,
                semantic_guidance_object_positions=overall_object_positions,
                semantic_guidance_kwargs=overall_semantic_guidance_kwargs,
                frozen_steps=frozen_steps,
                frozen_mask=frozen_mask,
            )

            print(
                f"Generation with spatial guidance from input latents and first {frozen_steps} steps frozen (directly from the composed latents input)"
            )
            print("Generation from composed latents (with semantic guidance)")

    utils.free_memory()

    return EasyDict(image=images[0], so_img_list=so_img_list, final_prompt=prompt)
