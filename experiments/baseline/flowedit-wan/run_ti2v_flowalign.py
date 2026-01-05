#!/usr/bin/env python3
"""
FlowAlign with TI2V-5B (WAN 2.2)

Uses the installed diffusers 0.36.0 which supports TI2V-5B VAE,
and patches flowalign method at runtime.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2 which has more free memory
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import imageio
import argparse
import omegaconf
from PIL import Image

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from torchvision import transforms
from torchvision.transforms import GaussianBlur


def load_video(file_path: str):
    """Load video frames as PIL images."""
    images = []
    vid = imageio.get_reader(file_path)
    fps = vid.get_meta_data()['fps']
    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)
    return images, fps


def prepare_first_frame_condition(pipe, image, num_frames, latent_height, latent_width, device, dtype, vae_on_gpu=False):
    """
    Prepare first-frame condition for TI2V (expand_timesteps) mode.

    Args:
        pipe: Pipeline with VAE
        image: First frame image tensor [B, C, H, W] normalized to [-1, 1]
        num_frames: Number of video frames
        latent_height: Height of latent space
        latent_width: Width of latent space
        device: Target device
        dtype: Target dtype
        vae_on_gpu: Whether VAE is already on GPU

    Returns:
        condition: VAE-encoded first frame [B, z_dim, F, H, W]
        first_frame_mask: Mask tensor [B, 1, F, H, W] where 0=condition, 1=latent
    """
    vae_scale_factor_temporal = 2 ** sum(pipe.vae.temperal_downsample)
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

    # Expand image to video format [B, C, 1, H, W]
    video_condition = image.unsqueeze(2)

    # Handle VAE device
    if not vae_on_gpu:
        pipe.vae.to(device)
    video_condition = video_condition.to(device=device, dtype=pipe.vae.dtype)

    # Encode with VAE
    with torch.no_grad():
        latent_condition = pipe.vae.encode(video_condition).latent_dist.sample()

    if not vae_on_gpu:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    # Normalize with VAE latent statistics
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        device, dtype
    )
    latent_condition = latent_condition.to(dtype)
    latent_condition = (latent_condition - latents_mean) * latents_std

    # Create first-frame mask (0 = use condition, 1 = use latent)
    first_frame_mask = torch.ones(
        1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
    )
    first_frame_mask[:, :, 0] = 0  # First frame uses condition

    return latent_condition, first_frame_mask


def ti2v_flowalign(
    pipe,
    video,
    source_prompt,
    target_prompt,
    target_image=None,
    height=480,
    width=832,
    num_inference_steps=50,
    strength=0.7,
    target_guidance_scale=19.5,
    fg_zeta_scale=0.01,
    bg_zeta_scale=0.01,
    source_idx=None,
    target_idx=None,
    device="cuda",
):
    """
    FlowAlign for TI2V-5B with first-frame conditioning.

    Simplified version focusing on the core FlowAlign algorithm with TI2V support.
    """
    transformer_dtype = pipe.transformer.dtype

    # Encode prompts
    print("Encoding prompts...")
    source_prompt_embeds = pipe.encode_prompt(
        source_prompt,
        do_classifier_free_guidance=False,
        device=device,
    )[0].to(transformer_dtype)

    target_prompt_embeds = pipe.encode_prompt(
        target_prompt,
        do_classifier_free_guidance=False,
        device=device,
    )[0].to(transformer_dtype)

    print(f"Prompt embeds shape: {source_prompt_embeds.shape}")

    # Preprocess video
    video_tensor = pipe.video_processor.preprocess_video(video, height=height, width=width).to(
        device, dtype=torch.float32
    )
    print(f"Video tensor shape: {video_tensor.shape}")

    # Encode video to latent space
    num_frames = video_tensor.shape[2]

    # Handle device properly for CPU offload
    # Move VAE to GPU for encoding
    vae_dtype = pipe.vae.dtype
    pipe.vae.to(device)

    with torch.no_grad():
        video_for_encode = video_tensor.to(device, dtype=vae_dtype)
        X0_src = pipe.vae.encode(video_for_encode).latent_dist.sample()

    # Move VAE back to CPU if using offload
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    # Normalize latent
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(X0_src.device, X0_src.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(X0_src.device, X0_src.dtype)
    X0_src = (X0_src - latents_mean) * latents_std
    X0_src = X0_src.to(device, dtype=transformer_dtype)

    print(f"Source latent shape: {X0_src.shape}")  # Should be [1, 48, F, H, W] for TI2V

    _, _, num_latent_frames, latent_height, latent_width = X0_src.shape

    # Prepare first-frame conditions (TI2V mode)
    expand_timesteps = getattr(pipe.config, 'expand_timesteps', False)
    print(f"expand_timesteps mode: {expand_timesteps}")

    source_condition = None
    target_condition = None
    first_frame_mask = None

    if expand_timesteps:
        print("[TI2V Mode] Preparing first-frame conditions...")

        # Source condition from video's first frame
        first_frame = video_tensor[:, :, 0, :, :].to(device)
        source_condition, first_frame_mask = prepare_first_frame_condition(
            pipe, first_frame, num_frames, latent_height, latent_width, device, transformer_dtype
        )
        print(f"Source condition shape: {source_condition.shape}")

        # Target condition from target image
        if target_image is not None:
            target_transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            target_frame = target_transform(target_image).unsqueeze(0).to(device)
            target_condition, _ = prepare_first_frame_condition(
                pipe, target_frame, num_frames, latent_height, latent_width, device, transformer_dtype
            )
            print(f"Target condition shape: {target_condition.shape}")
        else:
            target_condition = source_condition.clone()
            print("[TI2V Mode] No target image, using source first frame")

    # Setup timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    start_idx = int(len(timesteps) * (1 - strength))
    timesteps = timesteps[start_idx:]
    print(f"Timesteps: {len(timesteps)} steps, starting from step {start_idx}")

    # Initialize edit latent
    Zt_edit = X0_src.clone()

    # Gaussian blur for attention masking
    gaussian_blur = GaussianBlur(kernel_size=7, sigma=0.5)

    # Denoising loop
    print("Starting FlowAlign denoising loop...")
    for i, t in enumerate(timesteps):
        t_i = t / 1000.0

        # Forward diffusion on source
        fwd_noise = torch.randn_like(X0_src)
        Zt_src = (1 - t_i) * X0_src + t_i * fwd_noise

        # FlowAlign: target = edit + source - X0
        Zt_tar = Zt_edit + Zt_src - X0_src

        # Process samples sequentially to avoid OOM (5B model on 32GB GPU)
        # FlowAlign needs: vq_source (source with source prompt), vp_target (target with target prompt)

        # Prepare inputs for each forward pass
        inputs = [
            (Zt_src, source_condition, source_prompt_embeds, "vq_source"),
            (Zt_tar, target_condition, target_prompt_embeds, "vp_target"),
        ]

        predictions = []
        for latent, condition, prompt_embeds, name in inputs:
            latent_input = latent.clone()

            # Apply first-frame condition for TI2V mode
            if expand_timesteps and condition is not None:
                latent_input = (1 - first_frame_mask) * condition + first_frame_mask * latent_input

            # Prepare timestep
            if expand_timesteps:
                temp_ts = (first_frame_mask[0, 0, :, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0)
            else:
                timestep = t.unsqueeze(0)

            # Transformer forward with mixed precision (single sample)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                flow_pred = pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            predictions.append(flow_pred)
            # Clear cache after each forward to save memory
            torch.cuda.empty_cache()

        vq_source, vp_target = predictions

        # FlowAlign update
        dt = 1.0 / len(timesteps)

        # Update edit latent with zeta interpolation
        Zt_edit = Zt_edit - dt * (vp_target - vq_source)

        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{len(timesteps)}")

    # Decode final latent
    print("Decoding output video...")
    # Denormalize
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(Zt_edit.device, Zt_edit.dtype)
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(Zt_edit.device, Zt_edit.dtype)
    Zt_edit = Zt_edit / latents_std + latents_mean

    # Move VAE to GPU for decoding
    pipe.vae.to(device)
    Zt_edit = Zt_edit.to(device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        output_video = pipe.vae.decode(Zt_edit, return_dict=False)[0]
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    # Post-process
    output_video = pipe.video_processor.postprocess_video(output_video, output_type="np")[0]

    return output_video


def get_args():
    parser = argparse.ArgumentParser(description="FlowAlign with TI2V-5B")
    parser.add_argument('--config', type=str, default='./config/pvtt/bracelet_to_necklace_ti2v.yaml')
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    return config


if __name__ == '__main__':
    config = get_args()

    model_id = config.get('model_id', "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    print(f"[Model] Loading: {model_id}")

    # Load model
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.scheduler = scheduler
    # Use model CPU offload (allows manual VAE movement, unlike sequential offload)
    pipe.enable_model_cpu_offload()
    # Enable attention slicing to reduce memory usage
    pipe.enable_attention_slicing("max")

    print(f"expand_timesteps config: {getattr(pipe.config, 'expand_timesteps', 'NOT FOUND')}")

    # Load video (limit frames to avoid OOM)
    video, fps = load_video(config['video']['video_path'])
    max_frames = 9  # Limit to 9 frames for 5B model memory constraints
    if len(video) > max_frames:
        print(f"Limiting video from {len(video)} to {max_frames} frames")
        video = video[:max_frames]
    print(f"Video: {len(video)} frames, {fps} fps, {video[0].size}")

    # Load target image if specified
    target_image = None
    if config['video'].get('target_image'):
        target_image = Image.open(config['video']['target_image']).convert("RGB")
        print(f"Target image: {target_image.size}")

    # Run FlowAlign
    output = ti2v_flowalign(
        pipe=pipe,
        video=video,
        source_prompt=config['video']['source_prompt'],
        target_prompt=config['video']['target_prompt'],
        target_image=target_image,
        height=320,  # Reduced from 480 for memory
        width=576,   # Reduced from 832 for memory
        num_inference_steps=config['infernece']['num_inference_step'],
        strength=config['flowalign']['strength'],
        target_guidance_scale=config['flowalign']['target_guidance_scale'],
        fg_zeta_scale=config['flowalign']['zeta_scale'],
        bg_zeta_scale=config['flowalign']['bg_zeta_scale'],
    )

    # Save output
    save_path = config['flowalign']['save_video']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    export_to_video(output, save_path, fps=16)
    print(f"Output saved to: {save_path}")
    print("Done!")
