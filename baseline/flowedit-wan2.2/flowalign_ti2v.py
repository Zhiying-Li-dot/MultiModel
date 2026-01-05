#!/usr/bin/env python3
"""
FlowAlign with TI2V-5B using official Wan2.2 code.
Correctly implements the FlowAlign algorithm from the paper.
"""
import argparse
import gc
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from wan.utils.utils import save_video, masks_like, best_output_size


def load_video(file_path: str):
    """Load video frames as PIL images."""
    images = []
    vid = imageio.get_reader(file_path)
    fps = vid.get_meta_data()['fps']
    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)
    return images, fps


def flowalign_ti2v(
    model,
    vae,
    text_encoder,
    source_video,
    source_prompt,
    target_prompt,
    target_image=None,
    max_area=704 * 1280,
    frame_num=17,
    shift=5.0,
    sampling_steps=50,
    strength=0.7,
    target_guidance_scale=5.0,
    zeta_scale=0.01,
    n_prompt="",
    seed=-1,
    offload_model=True,
    device="cuda",
    param_dtype=torch.bfloat16,
):
    """
    FlowAlign for video editing with TI2V-5B.
    Implements the correct FlowAlign algorithm.
    """
    vae_stride = (4, 16, 16)
    patch_size = (1, 2, 2)
    num_train_timesteps = 1000

    # Get source video first frame for sizing
    source_first = source_video[0]
    ih, iw = source_first.height, source_first.width
    dh, dw = patch_size[1] * vae_stride[1], patch_size[2] * vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)
    print(f"Output size: {ow}x{oh}")

    # Limit frames
    F = min(frame_num, len(source_video))
    if F != len(source_video):
        print(f"Limiting frames from {len(source_video)} to {F}")
        source_video = source_video[:F]

    # Process frames
    def process_image(img, ow, oh):
        scale = max(ow / img.width, oh / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        return TF.to_tensor(img).sub_(0.5).div_(0.5)

    # Source video tensor [C, F, H, W]
    source_frames = [process_image(f, ow, oh) for f in source_video]
    source_video_tensor = torch.stack(source_frames, dim=1).to(device)

    # Source first frame [C, 1, H, W]
    source_first_tensor = source_video_tensor[:, :1, :, :]

    # Target first frame
    if target_image is not None:
        target_first_tensor = process_image(target_image, ow, oh).to(device).unsqueeze(1)
    else:
        target_first_tensor = source_first_tensor.clone()

    # Compute seq_len
    num_latent_frames = (F - 1) // vae_stride[0] + 1
    latent_h = oh // vae_stride[1]
    latent_w = ow // vae_stride[2]
    seq_len = num_latent_frames * latent_h * latent_w // (patch_size[1] * patch_size[2])
    print(f"Latent shape: [{vae.model.z_dim}, {num_latent_frames}, {latent_h}, {latent_w}], seq_len={seq_len}")

    # Encode prompts
    print("Encoding prompts...")
    text_encoder.model.to(device)
    context_source = text_encoder([source_prompt], device)
    context_target = text_encoder([target_prompt], device)
    if offload_model:
        text_encoder.model.cpu()
        torch.cuda.empty_cache()

    # Encode source video
    print("Encoding source video...")
    z_source = vae.encode([source_video_tensor])
    X0_src = z_source[0].to(device)  # [z_dim, F', H', W']
    print(f"Source latent shape: {X0_src.shape}")

    # Encode first frames for TI2V conditioning
    z_source_first = vae.encode([source_first_tensor])[0].to(device)
    z_target_first = vae.encode([target_first_tensor])[0].to(device)

    # Create masks
    mask1, mask2 = masks_like([X0_src], zero=True)
    mask2 = mask2[0].to(device)  # [z_dim, F', H', W'], first frame = 0

    # Setup scheduler
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
    timesteps = sample_scheduler.timesteps

    # Apply strength
    start_idx = int(len(timesteps) * (1 - strength))
    timesteps = timesteps[start_idx:]
    print(f"FlowAlign: {len(timesteps)} steps (strength={strength})")

    # Initialize
    Zt_edit = X0_src.clone()

    # Move model to GPU
    if offload_model:
        model.to(device)
        torch.cuda.empty_cache()

    # FlowAlign loop
    print("Starting FlowAlign loop...")
    with torch.amp.autocast('cuda', dtype=param_dtype), torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            t_i = t.item() / num_train_timesteps

            # Get t_{i-1}
            if i + 1 < len(timesteps):
                t_im1 = timesteps[i + 1].item() / num_train_timesteps
            else:
                t_im1 = 0.0

            # Forward diffusion on source (new noise each step, same as diffusers)
            noise = torch.randn(X0_src.shape, device=device, dtype=torch.float32)
            Zt_src = (1 - t_i) * X0_src + t_i * noise

            # FlowAlign coupling
            Zt_tar = Zt_edit + (Zt_src - X0_src)

            # Apply first-frame conditioning for TI2V
            # CRITICAL: vp_source uses SOURCE condition, vp_target uses TARGET condition
            # Build latent batch: [Zt_src, Zt_tar, Zt_tar]
            latent_batch = torch.stack([Zt_src, Zt_tar, Zt_tar], dim=0)

            # Build condition batch: [source, source, target] - NOT [source, target, target]!
            condition_batch = torch.stack([z_source_first, z_source_first, z_target_first], dim=0)

            # Apply mask: first frame uses condition, other frames use latent
            concat_input = (1. - mask2) * condition_batch + mask2 * latent_batch

            # Prompt batch: [source, source, target]
            concat_context = [
                context_source[0],  # vq_source
                context_source[0],  # vp_source
                context_target[0],  # vp_target
            ]

            # Prepare timestep (expand_timesteps mode)
            temp_ts = (mask2[0, :, ::2, ::2] * t).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t])
            timestep = temp_ts.unsqueeze(0).expand(3, -1)

            # Model forward (batched)
            concat_flow_pred = model(
                [concat_input[0], concat_input[1], concat_input[2]],
                t=timestep,
                context=concat_context,
                seq_len=seq_len
            )

            vq_source = concat_flow_pred[0]
            vp_source = concat_flow_pred[1]
            vp_target = concat_flow_pred[2]

            if offload_model:
                torch.cuda.empty_cache()

            # CFG-style mixing for target
            vp = vp_source + target_guidance_scale * (vp_target - vp_source)
            vq = vq_source

            # FlowAlign update (correct formula from paper)
            # Zt_edit = Zt_edit + (t_im1 - t_i) * (vp - vq) + zeta_scale * (Zt_src - t_i*vq - Zt_tar + t_i*vp)
            Zt_edit = Zt_edit.to(torch.float32)
            Zt_edit = Zt_edit + (t_im1 - t_i) * (vp - vq) + zeta_scale * (Zt_src - t_i * vq - Zt_tar + t_i * vp)

            del concat_input, vq_source, vp_source, vp_target

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
    parser = argparse.ArgumentParser(description="FlowAlign with TI2V-5B")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--source_prompt", type=str, required=True)
    parser.add_argument("--target_prompt", type=str, required=True)
    parser.add_argument("--target_image", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--zeta_scale", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    target_image = None
    if args.target_image:
        target_image = Image.open(args.target_image).convert("RGB")
        print(f"Loaded target image: {target_image.size}")

    output = flowalign_ti2v(
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        source_video=source_video,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        target_image=target_image,
        max_area=704 * 1280,
        frame_num=args.frame_num,
        strength=args.strength,
        target_guidance_scale=args.guidance_scale,
        zeta_scale=args.zeta_scale,
        sampling_steps=args.steps,
        seed=args.seed,
        offload_model=True,
        device=device,
        param_dtype=config.param_dtype,
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
