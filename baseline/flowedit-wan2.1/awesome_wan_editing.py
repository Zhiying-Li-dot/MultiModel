# 1) define
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import imageio
import requests
from PIL import Image
import argparse
import omegaconf

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline # pipeline_wan.py
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor

from utils.wan_attention import register_attention_processor
from utils.reference_utils import extract_clean_reference_features, prepare_reference_latent

# 1) load video
# helper function to load videos
def load_video(file_path: str):
    images = []

    # Assuming it's a local file path
    vid = imageio.get_reader(file_path)

    fps = vid.get_meta_data()['fps']

    # Load all frames and take first 49 frames to match original setup
    all_frames = []
    for frame in vid:
        all_frames.append(frame)

    # Take first 49 frames (same as horsejump-high example)
    frames_to_use = all_frames[:49]
    print(f"[Video] Total frames: {len(all_frames)}, Using first 49 frames")

    for frame in frames_to_use:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images, fps

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a Training-Free Editing script for WAN2.1.")

    parser.add_argument('--config', type=str, default='./config/object_editing/bear_edit.yaml')
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)
    return config

if __name__ == '__main__':

    config = get_args()

    # Available models:
    # - Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers (T2V)
    # - Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (I2V with CLIP - architecture incompatible with FlowAlign)
    # - Wan-AI/Wan2.2-TI2V-5B-Diffusers (TI2V with VAE-based image conditioning - RECOMMENDED)

    # Check if target_image is specified and determine model type
    use_image_conditioning = config['video'].get('target_image', None) is not None
    model_id = config.get('model_id', None)

    # Determine which model to use
    if model_id is None:
        if use_image_conditioning:
            # Default to TI2V-5B for image conditioning (VAE-based, compatible with FlowAlign)
            model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        else:
            # Default to T2V-1.3B for text-only
            model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    print(f"[Model] Loading: {model_id}")

    # Check if it's a TI2V model (uses expand_timesteps, no CLIP encoder)
    is_ti2v = "TI2V" in model_id or "2.2" in model_id

    if is_ti2v:
        # TI2V model: Uses VAE-based image conditioning (expand_timesteps mode)
        print("[TI2V Mode] Using VAE-based first-frame conditioning")
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        flow_shift = 3.0  # 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16  # TI2V-5B works well with bfloat16
        )
    elif "I2V" in model_id:
        # I2V model: Uses CLIP-based image conditioning (NOT recommended - architecture incompatible)
        print("[I2V Mode] WARNING: I2V architecture may be incompatible with FlowAlign (36ch vs 16ch)")
        image_encoder = CLIPVisionModel.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=torch.float32
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            model_id, subfolder="image_processor"
        )
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        flow_shift = 3.0
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            image_encoder=image_encoder,
            image_processor=image_processor,
            torch_dtype=torch.float16
        )
    else:
        # T2V model: Text-only (original behavior)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)

    pipe.scheduler = scheduler

    # GPU memory management
    if is_ti2v:
        # TI2V-5B fits on single 32GB GPU
        pipe.enable_model_cpu_offload()
        print("[TI2V Mode] Using model CPU offload")
    elif "14B" in model_id:
        pipe.enable_sequential_cpu_offload()
        print("[Large Model] Using sequential CPU offload for 14B model")
    else:
        pipe.to("cuda")

    # load video
    video, fps = load_video(config['video']['video_path']) # horsejump-high

    print(len(video)) # 49
    print(fps) # 16.0
    print(video[0].size) # (832, 480)

    ########################################### prompt
    source_prompt = config['video']['source_prompt']
    target_prompt = config['video']['target_prompt']

    source_blend = config['video']['source_blend'] # foreground object
    target_blend = config['video']['target_blend'] # foreground object

    ## source prompt
    source_blend_idx = pipe.tokenizer(source_blend,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids
    #source_blend_embed = pipe.text_encoder(source_blend_idx.cuda()).last_hidden_state # (1, 512, 4096)
    #print(source_blend_idx)
    print(source_blend_idx.shape) # (1, 512)
    source_blend_idx = source_blend_idx[0]
    #print(pipe.tokenizer.decode(source_blend_idx[0]))
    #print(torch.nonzero(source_blend_idx, as_tuple=True))
    source_blend_idx = source_blend_idx[torch.nonzero(source_blend_idx, as_tuple=True)][:-1]
    print(pipe.tokenizer.decode(source_blend_idx), '//', source_blend_idx)

    source_prompt_idx = pipe.tokenizer(source_prompt,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids[0]
    #print(source_prompt_idx)
    idx_list = torch.where(source_prompt_idx==source_blend_idx[0])[0]
    for idx in idx_list:
        boolean = torch.all(source_prompt_idx[idx:idx+len(source_blend_idx)]==source_blend_idx)

        if boolean:
            source_idx = list(range(idx, idx+len(source_blend_idx)))
            break
    print(source_idx)

    ## target prompt
    target_blend_idx = pipe.tokenizer(target_blend,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids
    #print(target_blend_idx)
    print(target_blend_idx.shape) # (1, 512)
    target_blend_idx = target_blend_idx[0]
    target_blend_idx = target_blend_idx[torch.nonzero(target_blend_idx, as_tuple=True)][:-1]
    print(pipe.tokenizer.decode(target_blend_idx), '//', target_blend_idx)

    target_prompt_idx = pipe.tokenizer(target_prompt,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids[0]
    #print(target_prompt_idx)
    idx_list = torch.where(target_prompt_idx==target_blend_idx[0])[0]
    for idx in idx_list:
        boolean = torch.all(target_prompt_idx[idx:idx+len(target_blend_idx)]==target_blend_idx)

        if boolean:
            target_idx = list(range(idx, idx+len(target_blend_idx)))
            break
    print(target_idx)
    ###########################################

    # inference
    if config['training-free-type']['flag_flowalign']:

        # model register
        ## hyperparameters
        num_inference_steps = config['infernece']['num_inference_step']
        strength = config['flowalign']['strength']
        total_steps = int(num_inference_steps*strength)
        target_guidance_scale = config['flowalign']['target_guidance_scale']
        fg_zeta_scale = config['flowalign']['zeta_scale'] # foreground (attn-masking) or all (no attn-masking)
        bg_zeta_scale = config['flowalign']['bg_zeta_scale'] # background
        num_frames = len(video)
        print("Total steps:", total_steps)

        start_step = 1
        end_step = total_steps
        start_layer = 1
        end_layer = 30
        layer_idx = None
        total_layers = 30

        masking_flag = config['flowalign']['flag_attnmask']
        masking_layer = [11, 12, 13, 14, 15, 16, 17]

        # Load target image if specified (for Reference Attention or TI2V)
        target_image = None
        if config['video'].get('target_image', None):
            target_image_path = config['video']['target_image']
            target_image = Image.open(target_image_path).convert("RGB")
            print(f"[Image] Loaded target image: {target_image_path}")

        # Check if using Reference Self-Attention
        use_reference_attention = config['flowalign'].get('use_reference_attention', False)

        ## Replace attention processor
        # AttributeError: 'WanTransformer3DModel' object has no attribute 'set_attn_processor'
        # Be careful, kyujinpy modified WanTransformer3DModel
        # Add `set_attn_processor` and `attn_processor` functions in above class.

        if use_reference_attention and target_image is not None:
            # Use Clean Reference Attention (RefDrop PVTT Adaptation)
            print("[Clean RefDrop] Starting reference attention setup...")

            # Get reference parameters from config
            ref_c = config['flowalign'].get('ref_c', 0.2)  # Default 0.2 (RefDrop video recommendation)

            # Extract clean reference features (at t=0)
            ref_bank = extract_clean_reference_features(
                product_image=target_image,
                vae=pipe.vae,
                transformer=pipe.transformer,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                ref_prompt=target_prompt,  # Use target prompt as reference context
                target_size=(480, 832),
                device=pipe.device
            )

            # Register Clean Reference Attention Processor
            register_attention_processor(
                pipe.transformer,
                processor_type="CleanReferenceAttentionProcessor",
                ref_bank=ref_bank,
                c=ref_c
            )

            print(f"[Clean RefDrop] Registered (c={ref_c})")

        else:
            # Use MasaCtrl (original behavior)
            register_attention_processor(pipe.transformer,
                                        processor_type="MasaCtrlProcessor",
                                        start_step=start_step,
                                        end_step=end_step,
                                        start_layer=start_layer, # attention layer
                                        end_layer=end_layer,
                                        layer_idx=layer_idx,
                                        total_layers=total_layers, # WAN2.1-1.3B has 30 layers / 14B: ??
                                        total_steps=total_steps,
                                        masking=masking_flag, # apply cross-attn masking
                                        masking_layer=masking_layer,
                                        frame_num=num_frames,
                                        source_idx=source_idx,
                                        target_idx=target_idx,
                                        )

            if use_reference_attention and target_image is None:
                print("[Warning] use_reference_attention=True but no target_image provided, using MasaCtrl")

        output = pipe.flowalign(
            video = video,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            target_image=target_image,  # NEW: pass target image for conditioning
            height=480,
            width=832,
            num_inference_steps=num_inference_steps,
            strength=strength,
            target_guidance_scale=target_guidance_scale, # [5.0, 7.5, 10.0, 13.5]
            fg_zeta_scale=fg_zeta_scale,
            bg_zeta_scale=bg_zeta_scale,
        ).frames[0]

        export_to_video(output, config['flowalign']['save_video'], fps=16)

        print(output.shape)
        print("WANAlign2.1 Finish!!")
    
    elif config['training-free-type']['flag_flowedit']:

        # model register
        ## hyperparameters
        num_inference_steps = config['infernece']['num_inference_step']
        strength = config['flowedit']['strength']
        total_steps = int(num_inference_steps*strength)
        target_guidance_scale = config['flowedit']['target_guidance_scale']
        source_guidance_scale = config['flowedit']['source_guidance_scale']
        num_frames = len(video)
        print("Total steps:", total_steps)

        # Load target image if specified (for Reference Attention)
        target_image = None
        if config['video'].get('target_image', None):
            target_image_path = config['video']['target_image']
            target_image = Image.open(target_image_path).convert("RGB")
            print(f"[Image] Loaded target image: {target_image_path}")

        # Check if using Reference Self-Attention
        use_reference_attention = config['flowedit'].get('use_reference_attention', False)
        use_noisy_refdrop = config['flowedit'].get('noisy_refdrop', False)  # NEW: noisy mode

        if use_reference_attention and target_image is not None:
            # Get reference parameters from config
            ref_c = config['flowedit'].get('ref_c', 0.2)

            if use_noisy_refdrop:
                # NEW: Noisy RefDrop - pre-compute features for all timesteps
                print("[Noisy RefDrop] Starting reference attention setup for FlowEdit...")

                # Prepare reference latent
                ref_data = prepare_reference_latent(
                    product_image=target_image,
                    vae=pipe.vae,
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    ref_prompt=target_prompt,
                    target_size=(480, 832),
                    device=pipe.device
                )

                # Compute timesteps (same logic as FlowEdit pipeline)
                pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
                timesteps = pipe.scheduler.timesteps
                # Apply strength to get actual timesteps used
                init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                t_start = max(num_inference_steps - init_timestep, 0)
                timesteps = timesteps[t_start:]
                print(f"[Noisy RefDrop] Timesteps: {len(timesteps)} steps, range [{timesteps[0].item()}, {timesteps[-1].item()}]")

                # Register Noisy Reference Attention Processor with pre-computed features
                register_attention_processor(
                    pipe.transformer,
                    processor_type="NoisyReferenceAttentionProcessor",
                    ref_latent=ref_data["ref_latent"],
                    prompt_embeds=ref_data["prompt_embeds"],
                    transformer=pipe.transformer,
                    scheduler=pipe.scheduler,
                    timesteps=timesteps,
                    c=ref_c
                )

                print(f"[Noisy RefDrop] Registered for FlowEdit (c={ref_c})")

            else:
                # Original: Clean RefDrop - fixed features from t=0
                print("[Clean RefDrop] Starting reference attention setup for FlowEdit...")

                # Extract clean reference features (at t=0)
                ref_bank = extract_clean_reference_features(
                    product_image=target_image,
                    vae=pipe.vae,
                    transformer=pipe.transformer,
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    ref_prompt=target_prompt,
                    target_size=(480, 832),
                    device=pipe.device
                )

                # Register Clean Reference Attention Processor
                register_attention_processor(
                    pipe.transformer,
                    processor_type="CleanReferenceAttentionProcessor",
                    ref_bank=ref_bank,
                    c=ref_c
                )

                print(f"[Clean RefDrop] Registered for FlowEdit (c={ref_c})")

        else:
            # Use default attention processor
            register_attention_processor(pipe.transformer,
                                        processor_type="WanAttnProcessor2_0",
                                        )
            if use_reference_attention and target_image is None:
                print("[Warning] use_reference_attention=True but no target_image provided")

        output = pipe.flowedit(
            video = video,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            height=480,
            width=832,
            num_inference_steps=num_inference_steps,
            strength=strength,
            target_guidance_scale=target_guidance_scale,
            source_guidance_scale=source_guidance_scale,
        ).frames[0]

        export_to_video(output, config['flowedit']['save_video'], fps=16)

        print(output.shape)
        print("WANEdit2.1 Finish!!")
    
    else:
        raise Exception("Not support.")