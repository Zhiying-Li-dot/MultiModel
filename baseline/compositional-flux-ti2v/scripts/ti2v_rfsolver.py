#!/usr/bin/env python3
"""
RF-Solver + TI2V

Uses RF-Solver (second-order Taylor expansion) for more accurate inversion:
1. Inversion with RF-Solver: Run ODE backwards (t: 0→1) with second-order accuracy
2. TI2V: Replace first frame with target, denoise (t: 1→0) to generate target video

RF-Solver Algorithm:
    For each step from t to t+dt:
    1. Compute v = model(z_t, t)
    2. Half step: z_mid = z_t + v * (dt/2)
    3. Compute v_mid = model(z_mid, t + dt/2)
    4. Estimate dv/dt = 2 * (v_mid - v) / dt
    5. Second-order update: z_{t+dt} = z_t + v * dt + 0.5 * dt² * (dv/dt)

Reference: "Taming Rectified Flow for Inversion and Editing" (arXiv:2411.04746)
"""
import argparse
import gc
import logging
import os
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import imageio
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.fm_solvers import get_sampling_sigmas
from wan.utils.utils import masks_like, save_video

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)


def load_video(file_path: str, max_frames: int = 49):
    """Load video frames as PIL images."""
    vid = imageio.get_reader(file_path)
    fps = vid.get_meta_data()['fps']

    frames = []
    for i, frame in enumerate(vid):
        if i >= max_frames:
            break
        frames.append(Image.fromarray(frame))

    logging.info(f"[Video] Loaded {len(frames)} frames, fps={fps}")
    return frames, fps


def process_image(img, ow, oh):
    """Resize with LANCZOS and center crop."""
    scale = max(ow / img.width, oh / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    return TF.to_tensor(img).sub_(0.5).div_(0.5)


def ti2v_rfsolver(
    checkpoint_dir: str,
    source_video_path: str,
    target_frame_path: str,
    source_prompt: str,
    target_prompt: str,
    output_path: str,
    max_frames: int = 49,
    sampling_steps: int = 50,
    guide_scale: float = 5.0,
    shift: float = 0.5,
    denoise_shift: float = None,  # If None, use same as shift
    device_id: int = 0,
    target_width: int = None,
    target_height: int = None,
    seed: int = 42,
    use_rfsolver_denoise: bool = False,
    skip_inversion: bool = False,
):
    """
    RF-Solver Inversion + TI2V.

    Pipeline:
        1. Encode source video → z0_src
        2. RF-Solver Inversion (t: 0→1) → zT (noise) with second-order accuracy
        3. Replace first frame with target first frame
        4. TI2V denoising (t: 1→0) → target video

    Args:
        checkpoint_dir: Path to Wan2.2-TI2V-5B checkpoint
        source_video_path: Source video file
        target_frame_path: Target first frame (from Flux.2)
        source_prompt: Description of source video
        target_prompt: Description of target video
        output_path: Output video path
        max_frames: Maximum frames to process
        sampling_steps: Number of ODE steps
        guide_scale: CFG scale for denoising
        shift: Flow matching shift parameter
        device_id: CUDA device
        seed: Random seed
        use_rfsolver_denoise: Whether to use RF-Solver for denoising too
    """
    device = torch.device(f"cuda:{device_id}")

    # Load config
    cfg = WAN_CONFIGS["ti2v-5B"]
    param_dtype = cfg.param_dtype
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size
    num_train_timesteps = cfg.num_train_timesteps

    logging.info(f"[Config] param_dtype={param_dtype}, vae_stride={vae_stride}")

    # Set random seed
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"[Seed] {seed}")

    # Load models
    logging.info("[1/7] Loading T5 text encoder...")
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
    )

    logging.info("[2/7] Loading VAE...")
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(checkpoint_dir, cfg.vae_checkpoint),
        device=device
    )

    logging.info("[3/7] Loading DiT model...")
    model = WanModel.from_pretrained(checkpoint_dir)
    model.eval().requires_grad_(False)

    # Load source video
    logging.info("[4/7] Loading source video...")
    source_frames, fps = load_video(source_video_path, max_frames)
    F = len(source_frames)
    ih, iw = source_frames[0].height, source_frames[0].width
    logging.info(f"  Video: {F} frames, {iw}x{ih}")

    # Compute output size
    from wan.utils.utils import best_output_size
    dh, dw = patch_size[1] * vae_stride[1], patch_size[2] * vae_stride[2]
    max_area = 704 * 1280
    if target_width and target_height:
        target_w = (target_width // dw) * dw
        target_h = (target_height // dh) * dh
        logging.info(f"  Output size: {target_w}x{target_h} (forced)")
    else:
        target_w, target_h = best_output_size(iw, ih, dw, dh, max_area)
        logging.info(f"  Output size: {target_w}x{target_h} (auto)")

    # Process frames
    source_tensors = [process_image(f, target_w, target_h) for f in source_frames]
    source_video = torch.stack(source_tensors, dim=1).to(device)  # [C, F, H, W]
    logging.info(f"  Processed video: {source_video.shape}")

    # Load target first frame
    logging.info("[5/7] Loading target first frame...")
    target_img = Image.open(target_frame_path).convert("RGB")
    target_frame = process_image(target_img, target_w, target_h).to(device)  # [C, H, W]
    logging.info(f"  Target frame: {target_frame.shape}")

    # Encode prompts
    logging.info("[6/7] Encoding prompts...")
    text_encoder.model.to(device)
    source_context = text_encoder([source_prompt], device)
    target_context = text_encoder([target_prompt], device)
    null_context = text_encoder([""], device)
    text_encoder.model.cpu()
    torch.cuda.empty_cache()

    # Compute latent dimensions
    latent_f = (F - 1) // vae_stride[0] + 1
    latent_h = target_h // vae_stride[1]
    latent_w = target_w // vae_stride[2]
    z_dim = vae.model.z_dim
    seq_len = latent_f * latent_h * latent_w // (patch_size[1] * patch_size[2])

    logging.info(f"  Latent shape: ({z_dim}, {latent_f}, {latent_h}, {latent_w})")
    logging.info(f"  Seq len: {seq_len}")

    # Encode source video
    logging.info("[7/7] Encoding source video...")
    z0_src = vae.encode([source_video])[0]  # [z_dim, latent_f, latent_h, latent_w]
    logging.info(f"  z0_src shape: {z0_src.shape}")

    # Encode source first frame (for TI2V conditioning during inversion)
    source_first = source_video[:, :1, :, :]  # [C, 1, H, W]
    z_source_first = vae.encode([source_first])[0]  # [z_dim, 1, latent_h, latent_w]
    logging.info(f"  z_source_first shape: {z_source_first.shape}")

    # Encode target first frame (for TI2V conditioning during generation)
    target_first = target_frame.unsqueeze(1)  # [C, 1, H, W]
    z_target_first = vae.encode([target_first])[0]  # [z_dim, 1, latent_h, latent_w]
    logging.info(f"  z_target_first shape: {z_target_first.shape}")

    # Create mask for TI2V (first frame = 0, others = 1)
    mask1, mask2 = masks_like([z0_src], zero=True)

    # Move model to GPU
    model.to(device)
    torch.cuda.empty_cache()

    @contextmanager
    def noop_no_sync():
        yield
    no_sync = getattr(model, 'no_sync', noop_no_sync)

    # Get sigmas for later use
    sigmas = get_sampling_sigmas(sampling_steps, shift)

    # ========== Step 1: RF-Solver Inversion (t: 0 → 1) or Random Noise ==========
    logging.info("=" * 50)
    if skip_inversion:
        logging.info("Step 1: SKIP Inversion - Using Random Noise")
        logging.info("=" * 50)
        # Use random noise instead of inversion
        zT = torch.randn_like(z0_src)
        logging.info(f"  Random noise zT: mean={zT.mean().item():.4f}, std={zT.std().item():.4f}")
    else:
        logging.info("Step 1: RF-Solver Inversion (t: 0 → 1)")
        logging.info("=" * 50)

        # Flip sigmas for inversion
        sigmas_inv = np.flip(sigmas).copy()  # [sigma_min, ..., sigma_max]

        logging.info(f"  Inversion steps: {len(sigmas_inv) - 1}")
        logging.info(f"  Sigmas: {sigmas_inv[0]:.4f} → {sigmas_inv[-1]:.4f}")
        logging.info(f"  Using RF-Solver (second-order)")

        # RF-Solver Inversion loop (with TI2V first-frame conditioning)
        zT = z0_src.clone()
        logging.info(f"  Using TI2V first-frame conditioning during inversion (lizhiying's method)")

        with (
            torch.amp.autocast('cuda', dtype=param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            for i in tqdm(range(len(sigmas_inv) - 1), desc="RF-Solver Inversion"):
                t_curr = sigmas_inv[i]
                t_next = sigmas_inv[i + 1]
                dt = t_next - t_curr  # dt > 0 for inversion

                # Apply TI2V first-frame conditioning with SOURCE first frame
                # Key insight from lizhiying: this ensures inversion properly encodes motion
                latent_input = (1. - mask2[0]) * z_source_first + mask2[0] * zT.to(param_dtype)

                # TI2V-style timestep: first frame t=0, others t=t
                t_val = t_curr * num_train_timesteps
                temp_ts = (mask2[0][0][:, ::2, ::2] * t_val).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_val])
                timestep_curr = temp_ts.unsqueeze(0).to(device)

                # Step 1: Compute v at current position
                v = model([latent_input], t=timestep_curr, context=source_context, seq_len=seq_len)[0]

                # Step 2: Half step to midpoint
                zT_float = zT.to(torch.float32)
                z_mid = zT_float + v.to(torch.float32) * (dt / 2)

                # Note: Do NOT apply TI2V conditioning at midpoint
                # Midpoint is only for derivative estimation, not actual latent state

                # Timestep for model (midpoint)
                t_mid = t_curr + dt / 2
                t_mid_val = t_mid * num_train_timesteps
                temp_ts_mid = (mask2[0][0][:, ::2, ::2] * t_mid_val).flatten()
                temp_ts_mid = torch.cat([temp_ts_mid, temp_ts_mid.new_ones(seq_len - temp_ts_mid.size(0)) * t_mid_val])
                timestep_mid = temp_ts_mid.unsqueeze(0).to(device)

                # Step 3: Compute v at midpoint (without TI2V conditioning on z_mid)
                v_mid = model([z_mid.to(param_dtype)], t=timestep_mid, context=source_context, seq_len=seq_len)[0]

                # Step 4: Estimate dv/dt using finite difference
                # dv/dt ≈ 2 * (v_mid - v) / dt
                dv_dt = 2 * (v_mid.to(torch.float32) - v.to(torch.float32)) / dt

                # Step 5: Second-order Taylor update
                # z_{t+dt} = z_t + v * dt + 0.5 * dt² * (dv/dt)
                zT = zT_float + v.to(torch.float32) * dt + 0.5 * dt * dt * dv_dt

                torch.cuda.empty_cache()

        logging.info(f"  Inverted noise zT: mean={zT.mean().item():.4f}, std={zT.std().item():.4f}")

    # ========== Step 2: TI2V Denoising (t: 1 → 0) ==========
    logging.info("=" * 50)
    logging.info("Step 2: TI2V Denoising (t: 1 → 0)")
    logging.info("=" * 50)

    # Setup scheduler for denoising (normal order)
    # Use denoise_shift if provided, otherwise use same shift as inversion
    actual_denoise_shift = denoise_shift if denoise_shift is not None else shift
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False
    )
    scheduler.set_timesteps(sampling_steps, device=device, shift=actual_denoise_shift)
    logging.info(f"  Inversion shift: {shift}, Denoising shift: {actual_denoise_shift}")

    logging.info(f"  Denoising steps: {len(scheduler.timesteps)}")
    logging.info(f"  CFG scale: {guide_scale}")
    logging.info(f"  Using {'RF-Solver' if use_rfsolver_denoise else 'Euler'} for denoising")

    # Replace first frame with target first frame
    latent = (1. - mask2[0]) * z_target_first + mask2[0] * zT
    logging.info(f"  Replaced first frame with target")

    # TI2V denoising loop
    with (
        torch.amp.autocast('cuda', dtype=param_dtype),
        torch.no_grad(),
        no_sync(),
    ):
        if use_rfsolver_denoise:
            # RF-Solver denoising
            sigmas_denoise = sigmas  # [sigma_max, ..., sigma_min]
            for i in tqdm(range(len(sigmas_denoise) - 1), desc="RF-Solver TI2V Denoising"):
                t_curr = sigmas_denoise[i]
                t_next = sigmas_denoise[i + 1]
                dt = t_next - t_curr  # dt < 0 for denoising

                # TI2V timestep: first frame t=0, others t=t
                temp_ts = (mask2[0][0][:, ::2, ::2] * t_curr * num_train_timesteps).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_curr * num_train_timesteps])
                timestep_curr = temp_ts.unsqueeze(0)

                # CFG at current position
                v_cond = model([latent], t=timestep_curr, context=target_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                v_uncond = model([latent], t=timestep_curr, context=null_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                v = v_uncond + guide_scale * (v_cond - v_uncond)

                # Half step
                latent_float = latent.to(torch.float32)
                latent_mid = latent_float + v.to(torch.float32) * (dt / 2)

                # Midpoint timestep
                t_mid = t_curr + dt / 2
                temp_ts_mid = (mask2[0][0][:, ::2, ::2] * t_mid * num_train_timesteps).flatten()
                temp_ts_mid = torch.cat([temp_ts_mid, temp_ts_mid.new_ones(seq_len - temp_ts_mid.size(0)) * t_mid * num_train_timesteps])
                timestep_mid = temp_ts_mid.unsqueeze(0)

                # CFG at midpoint
                v_mid_cond = model([latent_mid.to(latent.dtype)], t=timestep_mid, context=target_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                v_mid_uncond = model([latent_mid.to(latent.dtype)], t=timestep_mid, context=null_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                v_mid = v_mid_uncond + guide_scale * (v_mid_cond - v_mid_uncond)

                # Second-order update
                dv_dt = 2 * (v_mid.to(torch.float32) - v.to(torch.float32)) / dt
                latent = latent_float + v.to(torch.float32) * dt + 0.5 * dt * dt * dv_dt

                # Keep first frame fixed
                latent = (1. - mask2[0]) * z_target_first + mask2[0] * latent

                torch.cuda.empty_cache()
        else:
            # Standard Euler denoising (original)
            for i, t in enumerate(tqdm(scheduler.timesteps, desc="TI2V Denoising")):
                # TI2V timestep: first frame t=0, others t=t
                temp_ts = (mask2[0][0][:, ::2, ::2] * t).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t])
                timestep = temp_ts.unsqueeze(0)

                # CFG: v = v_uncond + scale * (v_cond - v_uncond)
                v_cond = model([latent], t=timestep, context=target_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                v_uncond = model([latent], t=timestep, context=null_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()

                v = v_uncond + guide_scale * (v_cond - v_uncond)

                # Scheduler step
                latent = latent.to(torch.float32)
                output = scheduler.step(v.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)
                latent = output[0].squeeze(0)

                # Keep first frame fixed
                latent = (1. - mask2[0]) * z_target_first + mask2[0] * latent

    # Move model to CPU
    model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Decode
    logging.info("Decoding output video...")
    output_video = vae.decode([latent])[0]  # [C, F, H, W]

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_video(
        tensor=output_video[None],  # [1, C, F, H, W]
        save_file=output_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    logging.info(f"Saved to: {output_path}")

    # Cleanup
    del model, text_encoder, vae
    gc.collect()
    torch.cuda.empty_cache()

    logging.info("RF-Solver + TI2V done!")


def main():
    parser = argparse.ArgumentParser(description="RF-Solver + TI2V")
    parser.add_argument("--checkpoint-dir", required=True, help="Wan2.2-TI2V-5B checkpoint")
    parser.add_argument("--source-video", required=True, help="Source video path")
    parser.add_argument("--target-frame", required=True, help="Target first frame (from Flux.2)")
    parser.add_argument("--source-prompt", required=True, help="Source video description")
    parser.add_argument("--target-prompt", required=True, help="Target video description")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale for denoising")
    parser.add_argument("--shift", type=float, default=0.5, help="Flow shift for inversion (0.5 recommended)")
    parser.add_argument("--denoise-shift", type=float, default=None, help="Flow shift for denoising (default: same as --shift)")
    parser.add_argument("--width", type=int, default=None, help="Force output width")
    parser.add_argument("--height", type=int, default=None, help="Force output height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rfsolver-denoise", action="store_true", help="Use RF-Solver for denoising too")
    parser.add_argument("--skip-inversion", action="store_true", help="Skip inversion, use random noise instead")

    args = parser.parse_args()

    ti2v_rfsolver(
        checkpoint_dir=args.checkpoint_dir,
        source_video_path=args.source_video,
        target_frame_path=args.target_frame,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        output_path=args.output,
        max_frames=args.max_frames,
        sampling_steps=args.steps,
        guide_scale=args.cfg,
        shift=args.shift,
        denoise_shift=args.denoise_shift,
        device_id=0,
        target_width=args.width,
        target_height=args.height,
        seed=args.seed,
        use_rfsolver_denoise=args.rfsolver_denoise,
        skip_inversion=args.skip_inversion,
    )


if __name__ == "__main__":
    main()
