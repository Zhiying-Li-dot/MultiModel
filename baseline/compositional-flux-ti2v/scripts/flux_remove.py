#!/usr/bin/env python3
"""
Flux.2 移除物体 - 用于两阶段编辑的 Stage 1

使用 FluxInpaintPipeline + mask 将图像中的指定物体移除。
"""

import argparse
from pathlib import Path

import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
from PIL import Image


def remove_object(
    input_image_path: str,
    mask_path: str,
    output_path: str,
    target_prompt: str,
    strength: float = 0.95,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
):
    """
    使用 Flux.2 inpainting 移除物体。

    Args:
        input_image_path: 输入图片路径
        mask_path: mask 图片路径（白色=要修复的区域）
        output_path: 输出图片路径
        target_prompt: 目标图片描述（不包含物体）
        strength: 编辑强度 (0-1)
        guidance_scale: Prompt 引导强度
        num_inference_steps: 推理步数
        seed: 随机种子
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    # 加载 Pipeline
    repo_id = "black-forest-labs/FLUX.1-dev"
    print(f"Loading model: {repo_id}")

    pipe = FluxInpaintPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
    )
    pipe.enable_sequential_cpu_offload(gpu_id=0)
    print("Pipeline loaded.")

    # 加载输入图片和 mask
    print(f"Loading input image: {input_image_path}")
    image = load_image(input_image_path)
    width, height = image.size

    print(f"Loading mask: {mask_path}")
    mask = load_image(mask_path).convert("L")
    # 确保 mask 和图片尺寸一致
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    # 生成图片
    print(f"Inpainting with strength={strength}, steps={num_inference_steps}")
    print(f"Target prompt: {target_prompt}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=target_prompt,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        strength=strength,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # 保存结果
    output_image = result.images[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    print(f"Saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Flux.2 移除物体 (Inpainting)")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--mask", required=True, help="mask 图片路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument("--target-prompt", required=True, help="目标图片描述（无物体）")
    parser.add_argument("--strength", type=float, default=0.95, help="编辑强度 (0-1)")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input image not found: {args.input}")
        return 1

    if not Path(args.mask).exists():
        print(f"Error: Mask image not found: {args.mask}")
        return 1

    remove_object(
        input_image_path=args.input,
        mask_path=args.mask,
        output_path=args.output,
        target_prompt=args.target_prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    print(f"\n✅ 完成！输出: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
