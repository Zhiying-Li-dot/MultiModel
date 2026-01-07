# 1) define
import os

# CRITICAL: Set CUDA device BEFORE any imports that might initialize CUDA
# This MUST be done before torch, transformers, diffusers are imported
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # fallback to GPU 0 if not set

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Debug: Print which GPU we're using
import sys
if len(sys.argv) > 1:
    print(f"[CUDA] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[CUDA] Will use GPU {os.environ['CUDA_VISIBLE_DEVICES']} (mapped to cuda:0)")

import torch

# Force PyTorch to respect CUDA_VISIBLE_DEVICES by setting device early
if torch.cuda.is_available():
    print(f"[CUDA] torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"[CUDA] torch.cuda.current_device() = {torch.cuda.current_device()}")
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        # When CUDA_VISIBLE_DEVICES=6, PyTorch sees it as device 0
        # So we explicitly set device 0 (which is the GPU we want)
        torch.cuda.set_device(0)
        print(f"[CUDA] Forced torch to use device 0 (physical GPU {os.environ['CUDA_VISIBLE_DEVICES']})")
        print(f"[CUDA] After set_device: torch.cuda.current_device() = {torch.cuda.current_device()}")

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

        ## replace attn processor
        # AttributeError: 'WanTransformer3DModel' object has no attribute 'set_attn_processor'
        # Be careful, kyujinpy modified WanTransformer3DModel
        # Add `set_attn_processor` and `attn_processor` functions in above class.
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
                                    
        # Load target image if specified (NEW: for image conditioning)
        target_image = None
        if config['video'].get('target_image', None):
            target_image_path = config['video']['target_image']
            target_image = Image.open(target_image_path).convert("RGB")
            print(f"[Image Conditioning] Loaded target image: {target_image_path}")

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

        register_attention_processor(pipe.transformer, 
                                    processor_type="WanAttnProcessor2_0",
                                    )
        
        # model register
        ## hyperparameters
        num_inference_steps = config['infernece']['num_inference_step']
        strength = config['flowedit']['strength']
        total_steps = int(num_inference_steps*strength)
        target_guidance_scale = config['flowedit']['target_guidance_scale']
        source_guidance_scale = config['flowedit']['source_guidance_scale']
        num_frames = len(video)
        print("Total steps:", total_steps)  

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