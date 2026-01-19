#!/usr/bin/env python3
"""
Stage 2: Wan2.1 TI2V - 视频生成

在 5090 服务器上运行，使用 Wan2.1 Image-to-Video 模型生成视频。
"""

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image


def generate_video(
    first_frame_path: str,
    prompt: str,
    output_path: str,
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
) -> str:
    """
    使用 Wan2.1 TI2V 从第一帧生成视频。

    Args:
        first_frame_path: 第一帧图片路径
        prompt: 视频描述
        output_path: 输出视频路径
        model_id: 模型 ID
        num_frames: 生成帧数
        num_inference_steps: 推理步数
        guidance_scale: 引导强度

    Returns:
        输出视频路径
    """
    print(f"Loading model: {model_id}")

    # 加载 VAE
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # 加载 Pipeline
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    # 使用 sequential CPU offload 减少显存占用
    pipe.enable_sequential_cpu_offload(gpu_id=0)

    # 加载第一帧
    print(f"Loading first frame: {first_frame_path}")
    image = Image.open(first_frame_path).convert("RGB")

    # 获取图片尺寸
    width, height = image.size
    print(f"Image size: {width}x{height}")

    # 生成视频
    print(f"Generating video with {num_frames} frames...")
    print(f"Prompt: {prompt}")

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt="blur, distort, low quality",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # 导出视频
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output.frames[0], output_path, fps=16)
    print(f"Saved video to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Wan2.1 TI2V - Stage 2")
    parser.add_argument("--first-frame", required=True, help="第一帧图片路径")
    parser.add_argument("--prompt", required=True, help="视频描述")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)

    args = parser.parse_args()

    if not Path(args.first_frame).exists():
        print(f"Error: First frame not found: {args.first_frame}")
        return 1

    generate_video(
        first_frame_path=args.first_frame,
        prompt=args.prompt,
        output_path=args.output,
        model_id=args.model_id,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    print("\n✅ Stage 2 完成！")
    return 0


if __name__ == "__main__":
    exit(main())
