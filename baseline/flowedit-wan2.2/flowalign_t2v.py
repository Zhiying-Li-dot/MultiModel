#!/usr/bin/env python3
"""
FlowAlign video editing with Wan2.2 TI2V-5B model (T2V mode).

This implementation uses pure text prompts for editing without first-frame conditioning.
Based on the FlowAlign algorithm, matching the diffusers flowedit-wan implementation.
"""
import argparse
import gc
import os
import sys
import random

import torch
import imageio
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import save_video, best_output_size


def load_video(file_path: str):
    """Load video frames as PIL images."""
    images = []
    vid = imageio.get_reader(file_path)
    fps = vid.get_meta_data()['fps']
    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)
    return images, fps


def flowedit_t2v(
    model,
    vae,
    text_encoder,
    source_video,
    source_prompt,
    target_prompt,
    max_area=704 * 1280,
    frame_num=17,
    shift=3.0,
    sampling_steps=50,
    strength=0.7,
    target_guidance_scale=13.5,  # FlowEdit default
    source_guidance_scale=5.0,   # FlowEdit uses source CFG too
    zeta_scale=0.001,            # Only used for FlowAlign
    method="flowalign",          # "flowedit" or "flowalign"
    seed=-1,
    offload_model=True,
    device="cuda",
    param_dtype=torch.bfloat16,
    target_width=None,
    target_height=None,
):
    """
    FlowEdit/FlowAlign for video editing using Wan2.2 TI2V-5B in T2V mode.

    Args:
        model: WanModel transformer
        vae: Wan2.2 VAE
        text_encoder: T5 text encoder
        source_video: List of PIL images
        source_prompt: Description of source video content
        target_prompt: Description of desired target content
        max_area: Maximum output resolution area
        frame_num: Number of frames to process
        shift: Flow matching shift parameter (3.0 matches diffusers)
        sampling_steps: Number of denoising steps
        strength: Edit strength (0.0-1.0), higher = more editing
        target_guidance_scale: CFG scale for target prompt
        source_guidance_scale: CFG scale for source prompt (FlowEdit only)
        zeta_scale: Structure preservation scale (FlowAlign only)
        method: "flowedit" or "flowalign"
        seed: Random seed
        offload_model: Whether to offload models to CPU when not in use
        device: Target device
        param_dtype: Model parameter dtype

    Returns:
        Edited video tensor [C, F, H, W]
    """
    print(f"Using method: {method}")
    vae_stride = (4, 16, 16)
    patch_size = (1, 2, 2)
    num_train_timesteps = 1000

    # Get output size from source video
    source_first = source_video[0]
    ih, iw = source_first.height, source_first.width
    dh, dw = patch_size[1] * vae_stride[1], patch_size[2] * vae_stride[2]
    # Use forced resolution or align to patch_size * vae_stride
    if target_width and target_height:
        ow = (target_width // dw) * dw
        oh = (target_height // dh) * dh
        print(f"Output size: {ow}x{oh} (forced)")
    else:
        ow = (iw // dw) * dw
        oh = (ih // dh) * dh
        print(f"Output size: {ow}x{oh} (original: {iw}x{ih})")

    # Limit frames
    F = min(frame_num, len(source_video))
    if F != len(source_video):
        print(f"Limiting frames from {len(source_video)} to {F}")
        source_video = source_video[:F]

    # Process frames to tensor
    def process_image(img, ow, oh):
        scale = max(ow / img.width, oh / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        return TF.to_tensor(img).sub_(0.5).div_(0.5)

    source_frames = [process_image(f, ow, oh) for f in source_video]
    source_video_tensor = torch.stack(source_frames, dim=1).to(device)

    # Compute sequence length for transformer
    num_latent_frames = (F - 1) // vae_stride[0] + 1
    latent_h = oh // vae_stride[1]
    latent_w = ow // vae_stride[2]
    seq_len = num_latent_frames * latent_h * latent_w // (patch_size[1] * patch_size[2])

    # Encode prompts
    print("Encoding prompts...")
    text_encoder.model.to(device)
    context_source = text_encoder([source_prompt], device)
    context_target = text_encoder([target_prompt], device)
    context_negative = text_encoder([""], device)  # negative prompt for CFG
    if offload_model:
        text_encoder.model.cpu()
        torch.cuda.empty_cache()

    # Encode source video to latent space
    print("Encoding source video...")
    z_source = vae.encode([source_video_tensor])
    X0_src = z_source[0].to(device)

    # Setup scheduler and timesteps
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
    timesteps = sample_scheduler.timesteps

    # Apply strength - skip early steps for partial editing
    start_idx = int(len(timesteps) * (1 - strength))
    timesteps = timesteps[start_idx:]
    print(f"FlowAlign: {len(timesteps)} steps (strength={strength})")

    # Initialize edit latent from source
    Zt_edit = X0_src.clone()

    # Set random seed for reproducibility and enable deterministic mode
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Move model to GPU
    if offload_model:
        model.to(device)
        torch.cuda.empty_cache()

    # Denoising loop
    with torch.amp.autocast('cuda', dtype=param_dtype), torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc=method)):
            t_i = t.item() / num_train_timesteps
            t_im1 = timesteps[i + 1].item() / num_train_timesteps if i + 1 < len(timesteps) else 0.0

            # Forward diffusion on source (new noise each step)
            noise = torch.randn(X0_src.shape, device=device, dtype=torch.float32)
            Zt_src = (1 - t_i) * X0_src + t_i * noise

            # Coupling: target shares noise with source
            Zt_tar = Zt_edit + (Zt_src - X0_src)

            if method == "flowedit":
                # FlowEdit: batch [Zt_src, Zt_tar] with CFG for both
                # Need 4 forward passes: src_uncond, tar_uncond, src_cond, tar_cond
                # Use negative prompt (empty string) for uncond, actual prompts for cond
                concat_context = [
                    context_negative[0],  # src uncond (negative prompt)
                    context_negative[0],  # tar uncond (negative prompt)
                    context_source[0],    # src cond (source prompt)
                    context_target[0],    # tar cond (target prompt)
                ]
                timestep = t.expand(seq_len).unsqueeze(0).expand(4, -1)

                concat_flow_pred = model(
                    [Zt_src, Zt_tar, Zt_src, Zt_tar],
                    t=timestep,
                    context=concat_context,
                    seq_len=seq_len
                )

                src_pred_uncond, tar_pred_uncond, src_pred_cond, tar_pred_cond = concat_flow_pred

                # CFG for both source and target
                Vt_src = src_pred_uncond + source_guidance_scale * (src_pred_cond - src_pred_uncond)
                Vt_tar = tar_pred_uncond + target_guidance_scale * (tar_pred_cond - tar_pred_uncond)

                # FlowEdit update: simple velocity difference
                Zt_edit = Zt_edit.to(torch.float32)
                V_delta = Vt_tar - Vt_src
                Zt_edit = Zt_edit + (t_im1 - t_i) * V_delta

            else:  # flowalign
                # FlowAlign: batch [Zt_src, Zt_tar, Zt_tar] with [src, src, tar] prompts
                concat_context = [context_source[0], context_source[0], context_target[0]]
                timestep = t.expand(seq_len).unsqueeze(0).expand(3, -1)

                concat_flow_pred = model(
                    [Zt_src, Zt_tar, Zt_tar],
                    t=timestep,
                    context=concat_context,
                    seq_len=seq_len
                )

                vq_source, vp_source, vp_target = concat_flow_pred

                # CFG-style mixing for target velocity
                vp = vp_source + target_guidance_scale * (vp_target - vp_source)
                vq = vq_source

                # FlowAlign update with structure preservation
                Zt_edit = Zt_edit.to(torch.float32)
                edit_term = (t_im1 - t_i) * (vp - vq)
                zeta_term = zeta_scale * (Zt_src - t_i * vq - Zt_tar + t_i * vp)
                Zt_edit = Zt_edit + edit_term + zeta_term

            if offload_model:
                torch.cuda.empty_cache()

    # Offload model
    if offload_model:
        model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Decode result
    print("Decoding output video...")
    output_video = vae.decode([Zt_edit])

    gc.collect()
    torch.cuda.synchronize()

    return output_video[0]


def main():
    parser = argparse.ArgumentParser(description="FlowEdit/FlowAlign video editing with Wan2.2")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Wan2.2-TI2V-5B checkpoint")
    parser.add_argument("--video", type=str, required=True, help="Source video path")
    parser.add_argument("--source_prompt", type=str, required=True, help="Description of source video")
    parser.add_argument("--target_prompt", type=str, required=True, help="Description of target video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--frame_num", type=int, default=17, help="Number of frames to process")
    parser.add_argument("--strength", type=float, default=0.7, help="Edit strength (0.0-1.0)")
    parser.add_argument("--method", type=str, default="flowalign", choices=["flowedit", "flowalign"], help="Editing method")
    parser.add_argument("--target_guidance_scale", type=float, default=None, help="Target guidance scale (default: 13.5 for flowedit, 19.5 for flowalign)")
    parser.add_argument("--source_guidance_scale", type=float, default=5.0, help="Source guidance scale (flowedit only)")
    parser.add_argument("--zeta_scale", type=float, default=0.001, help="Structure preservation scale (flowalign only)")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=None, help="Force output width")
    parser.add_argument("--height", type=int, default=None, help="Force output height")
    args = parser.parse_args()

    # Set default guidance scale based on method
    if args.target_guidance_scale is None:
        args.target_guidance_scale = 13.5 if args.method == "flowedit" else 19.5

    device = torch.device("cuda")
    config = WAN_CONFIGS["ti2v-5B"]

    print(f"Loading models from {args.ckpt_dir}...")

    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.ckpt_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, config.t5_tokenizer),
    )

    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.ckpt_dir, config.vae_checkpoint),
        device=device
    )

    model = WanModel.from_pretrained(args.ckpt_dir)
    model.eval().requires_grad_(False)

    print("Loading source video...")
    source_video, fps = load_video(args.video)
    print(f"Loaded {len(source_video)} frames at {fps} fps")

    output = flowedit_t2v(
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        source_video=source_video,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        max_area=704 * 1280,
        frame_num=args.frame_num,
        strength=args.strength,
        target_guidance_scale=args.target_guidance_scale,
        source_guidance_scale=args.source_guidance_scale,
        zeta_scale=args.zeta_scale,
        method=args.method,
        sampling_steps=args.steps,
        seed=args.seed,
        offload_model=True,
        device=device,
        param_dtype=config.param_dtype,
        target_width=args.width,
        target_height=args.height,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_video(
        tensor=output[None],
        save_file=args.output,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
