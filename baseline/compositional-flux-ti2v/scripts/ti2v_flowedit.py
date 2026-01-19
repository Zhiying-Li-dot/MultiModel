#!/usr/bin/env python3
"""
Stage 2: TI2V FlowEdit (using official wan package)

Combines FlowEdit algorithm with TI2V (first-frame) conditioning.
- Source branch: conditioned on source video's first frame
- Target branch: conditioned on Flux.2 generated first frame
"""
import argparse
import gc
import logging
import math
import os
import sys
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
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
from wan.utils.utils import masks_like

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
    """Resize with LANCZOS and center crop (same as flowedit-wan2.2)."""
    scale = max(ow / img.width, oh / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    return TF.to_tensor(img).sub_(0.5).div_(0.5)


def ti2v_flowedit(
    checkpoint_dir: str,
    source_video_path: str,
    target_frame_path: str,
    source_prompt: str,
    target_prompt: str,
    output_path: str,
    max_frames: int = 49,
    sampling_steps: int = 50,
    strength: float = 0.7,
    source_cfg: float = 5.0,
    target_cfg: float = 5.0,
    zeta_scale: float = 0.01,
    shift: float = 3.0,
    device_id: int = 0,
    use_image_cond: bool = True,
    target_width: int = None,
    target_height: int = None,
    seed: int = 42,
):
    """
    TI2V FlowEdit using official wan package.

    This implements FlowEdit algorithm with TI2V first-frame conditioning:
    - Source: Vt_src = CFG(V(Zt_src, source_prompt, source_condition))
    - Target: Vt_tar = CFG(V(Zt_tar, target_prompt, target_condition))
    - Update: Zt_edit += dt * (Vt_tar - Vt_src)

    Args:
        checkpoint_dir: Path to Wan2.2-TI2V-5B checkpoint
        source_video_path: Source video file
        target_frame_path: Flux.2 generated target first frame
        source_prompt: Description of source video
        target_prompt: Description of target video
        output_path: Output video path
        max_frames: Maximum frames to process
        sampling_steps: Denoising steps
        strength: Edit strength (0.7 = 30% of steps)
        source_cfg: CFG scale for source
        target_cfg: CFG scale for target
        shift: Flow matching shift
        device_id: CUDA device
    """
    device = torch.device(f"cuda:{device_id}")

    # Load config
    cfg = WAN_CONFIGS["ti2v-5B"]
    param_dtype = cfg.param_dtype
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size

    logging.info(f"[Config] param_dtype={param_dtype}, vae_stride={vae_stride}")

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

    # Use best_output_size for optimal resolution (same as flowedit-wan2.2)
    from wan.utils.utils import best_output_size
    dh, dw = patch_size[1] * vae_stride[1], patch_size[2] * vae_stride[2]
    max_area = 704 * 1280
    if target_width and target_height:
        # Use forced resolution (aligned to patch size)
        target_w = (target_width // dw) * dw
        target_h = (target_height // dh) * dh
        logging.info(f"  Output size: {target_w}x{target_h} (forced)")
    else:
        target_w, target_h = best_output_size(iw, ih, dw, dh, max_area)
        logging.info(f"  Output size: {target_w}x{target_h} (best_output_size)")

    # Process frames with LANCZOS + center crop (same as flowedit-wan2.2)
    source_tensors = [process_image(f, target_w, target_h) for f in source_frames]
    source_video = torch.stack(source_tensors, dim=1).to(device)  # [C, F, H, W]
    logging.info(f"  Processed video: {source_video.shape}")

    # Load target first frame (only if using image conditioning)
    if use_image_cond:
        logging.info("[5/7] Loading target first frame...")
        target_img = Image.open(target_frame_path).convert("RGB")
        target_frame = process_image(target_img, target_w, target_h).to(device)  # [C, H, W]
        logging.info(f"  Target frame: {target_frame.shape}")
    else:
        logging.info("[5/7] Skipping target first frame (no-image-cond mode)")
        target_frame = None

    # Encode prompts on GPU (same as flowalign_t2v.py for identical results)
    logging.info("[6/7] Encoding prompts...")
    text_encoder.model.to(device)
    source_context = text_encoder([source_prompt], device)
    target_context = text_encoder([target_prompt], device)
    null_context = text_encoder([""], device)  # empty string for CFG (same as flowalign_t2v.py)
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

    # Encode full source video for X0_src (same as flowalign_t2v.py)
    X0_src = vae.encode([source_video])[0]  # [z_dim, latent_f, latent_h, latent_w]
    logging.info(f"  X0_src shape: {X0_src.shape}")

    # Only encode first frames if using image conditioning
    if use_image_cond:
        source_first = source_video[:, :1, :, :]  # [C, 1, H, W]
        z_source = vae.encode([source_first])  # List of [z_dim, 1, latent_h, latent_w]
        target_first = target_frame.unsqueeze(1)  # [C, 1, H, W]
        z_target = vae.encode([target_first])
    else:
        z_source = None
        z_target = None

    # Create masks
    mask1, mask2 = masks_like([X0_src], zero=True)  # zero=True for i2v style

    # Setup scheduler
    logging.info("[7/7] Running FlowEdit denoising...")
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False
    )
    scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
    timesteps = scheduler.timesteps

    # Apply strength (skip early steps)
    start_idx = int(len(timesteps) * (1 - strength))
    timesteps = timesteps[start_idx:]
    logging.info(f"  {len(timesteps)} steps (strength={strength})")

    # Initialize edit latent from source
    Zt_edit = X0_src.clone()

    # Set random seed for reproducibility and enable deterministic mode (match flowalign_t2v.py exactly)
    import random
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Move model to GPU (after seed, same as flowalign_t2v.py)
    model.to(device)
    torch.cuda.empty_cache()

    @contextmanager
    def noop_no_sync():
        yield
    no_sync = getattr(model, 'no_sync', noop_no_sync)

    logging.info(f"  FlowEdit params: source_cfg={source_cfg}, target_cfg={target_cfg}, use_image_cond={use_image_cond}, seed={seed}")

    with (
        torch.amp.autocast('cuda', dtype=param_dtype),
        torch.no_grad(),
        no_sync(),
    ):
        num_train_timesteps = 1000
        for i, t in enumerate(tqdm(timesteps, desc="FlowAlign")):
            # Match flowalign_t2v.py exactly: use .item() to get Python float
            t_i = t.item() / num_train_timesteps
            t_im1 = timesteps[i + 1].item() / num_train_timesteps if i + 1 < len(timesteps) else 0.0

            # Forward diffusion on source (new noise each step)
            noise = torch.randn(X0_src.shape, device=device, dtype=torch.float32)
            Zt_src = (1 - t_i) * X0_src + t_i * noise

            # Compute target latent (for edit trajectory) - parentheses match flowalign_t2v.py
            Zt_tar = Zt_edit + (Zt_src - X0_src)

            # === FlowEdit 2-Branch Structure ===
            if use_image_cond:
                # TI2V mode: Apply first-frame conditioning
                Zt_src_cond = (1. - mask2[0]) * z_source[0] + mask2[0] * Zt_src
                Zt_tar_cond = (1. - mask2[0]) * z_target[0] + mask2[0] * Zt_tar
                # TI2V timestep: first frame t=0, others t=t
                temp_ts = (mask2[0][0][:, ::2, ::2] * t).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t])
                timestep = temp_ts.unsqueeze(0)

                # Separate forward passes for TI2V mode
                # Use null_context (empty string) for uncond (same as flowalign_t2v.py)
                src_pred_cond = model([Zt_src_cond], t=timestep, context=source_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                src_pred_uncond = model([Zt_src_cond], t=timestep, context=null_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                tar_pred_cond = model([Zt_tar_cond], t=timestep, context=target_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()
                tar_pred_uncond = model([Zt_tar_cond], t=timestep, context=null_context, seq_len=seq_len)[0]
                torch.cuda.empty_cache()

            else:
                # Standard FlowEdit: batched forward pass
                # Use negative prompt for uncond, actual prompts for cond (same as flowalign_t2v.py)
                timestep = t.expand(seq_len).unsqueeze(0).expand(4, -1)
                concat_context = [
                    null_context[0],    # src uncond (negative prompt)
                    null_context[0],    # tar uncond (negative prompt)
                    source_context[0],  # src cond (source prompt)
                    target_context[0],  # tar cond (target prompt)
                ]
                concat_flow_pred = model(
                    [Zt_src, Zt_tar, Zt_src, Zt_tar],
                    t=timestep,
                    context=concat_context,
                    seq_len=seq_len
                )
                src_pred_uncond, tar_pred_uncond, src_pred_cond, tar_pred_cond = concat_flow_pred

            # Apply CFG separately to source and target
            Vt_src = src_pred_uncond + source_cfg * (src_pred_cond - src_pred_uncond)
            Vt_tar = tar_pred_uncond + target_cfg * (tar_pred_cond - tar_pred_uncond)

            # FlowEdit update: Zt_edit += Δt * (Vt_tar - Vt_src)
            Zt_edit = Zt_edit.to(torch.float32)
            Zt_edit = Zt_edit + (t_im1 - t_i) * (Vt_tar - Vt_src)

            # Fix first frame to target condition (only in TI2V mode)
            if use_image_cond:
                Zt_edit = (1. - mask2[0]) * z_target[0] + mask2[0] * Zt_edit

    # Move model to CPU
    model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Decode
    logging.info("Decoding output video...")
    output_video = vae.decode([Zt_edit])[0]  # [C, F, H, W]

    # Save using official wan save_video (same as flowedit-wan2.2)
    from wan.utils.utils import save_video
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

    logging.info("✅ TI2V FlowEdit 完成!")


def main():
    parser = argparse.ArgumentParser(description="TI2V FlowAlign - Stage 2")
    parser.add_argument("--checkpoint-dir", required=True, help="Wan2.2-TI2V-5B checkpoint")
    parser.add_argument("--source-video", required=True, help="Source video path")
    parser.add_argument("--target-frame", required=True, help="Flux.2 generated first frame")
    parser.add_argument("--source-prompt", required=True, help="Source video description")
    parser.add_argument("--target-prompt", required=True, help="Target video description")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--source-cfg", type=float, default=5.0, help="(unused in FlowAlign)")
    parser.add_argument("--target-cfg", type=float, default=5.0, help="FlowAlign interpolation scale")
    parser.add_argument("--zeta-scale", type=float, default=0.01, help="Consistency regularization")
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--no-image-cond", action="store_true", help="Disable image conditioning (standard FlowEdit)")
    parser.add_argument("--width", type=int, default=None, help="Force output width (default: auto)")
    parser.add_argument("--height", type=int, default=None, help="Force output height (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    ti2v_flowedit(
        checkpoint_dir=args.checkpoint_dir,
        source_video_path=args.source_video,
        target_frame_path=args.target_frame,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        output_path=args.output,
        max_frames=args.max_frames,
        sampling_steps=args.steps,
        strength=args.strength,
        source_cfg=args.source_cfg,
        target_cfg=args.target_cfg,
        zeta_scale=args.zeta_scale,
        shift=args.shift,
        device_id=0,
        use_image_cond=not args.no_image_cond,
        target_width=args.width,
        target_height=args.height,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
