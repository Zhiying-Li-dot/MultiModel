#!/usr/bin/env python3
"""
Stage 1: Flux.2 Dev Edit - 图像编辑 (本地运行)

使用 Flux.2 Dev 模型将模板视频第一帧中的产品替换为目标产品。
支持多图参考编辑。
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from PIL import Image

# Prompt 模板
EDIT_PROMPT_TEMPLATE = """Replace the {source_object} from image 1 with the {target_object} from image 2.

Requirements:
- Keep the exact same camera angle and framing as image 1
- Preserve the lighting and shadows from image 1
- Maintain the background and surrounding elements from image 1
- The {target_object} should match the style, material, and details of image 2
- Ensure natural integration with proper scale and perspective
"""


def generate_target_first_frame(
    template_frame_path: str,
    product_image_path: str,
    source_object: str,
    target_object: str,
    output_path: str,
    guidance_scale: float = 2.5,
    num_inference_steps: int = 50,
    image_size: tuple = (832, 480),
    use_4bit: bool = False,
    seed: int = 42,
) -> str:
    """
    使用 Flux.2 Dev 生成目标视频第一帧。

    Args:
        template_frame_path: 模板视频第一帧路径
        product_image_path: 目标产品图片路径
        source_object: 源产品描述
        target_object: 目标产品描述
        output_path: 输出图片路径
        guidance_scale: Prompt 引导强度
        num_inference_steps: 推理步数
        image_size: 输出图片尺寸 (width, height)
        use_4bit: 是否使用 4-bit 量化
        seed: 随机种子

    Returns:
        输出图片路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    # 加载 Pipeline
    repo_id = "black-forest-labs/FLUX.2-dev"
    print(f"Loading model: {repo_id}")

    print("Loading Flux.2 pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
    )

    # 使用 sequential CPU offload 减少显存占用
    # 这会将模型的每一层在需要时加载到 GPU，用完后卸载回 CPU
    pipe.enable_sequential_cpu_offload(gpu_id=0)
    print("Pipeline loaded with sequential CPU offload.")

    # 加载参考图片
    print(f"Loading template frame: {template_frame_path}")
    image_1 = load_image(template_frame_path)

    print(f"Loading product image: {product_image_path}")
    image_2 = load_image(product_image_path)

    # 构建 prompt
    prompt = EDIT_PROMPT_TEMPLATE.format(
        source_object=source_object,
        target_object=target_object,
    )
    print(f"Prompt:\n{prompt}")

    # 生成图片
    print(f"Generating image with {num_inference_steps} steps...")
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=[image_1, image_2],
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=image_size[0],
        height=image_size[1],
    )

    # 保存结果
    output_image = result.images[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    print(f"Saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Flux.2 Dev Edit - Stage 1 (Local)")
    parser.add_argument("--template-frame", required=True, help="模板视频第一帧路径")
    parser.add_argument("--product-image", required=True, help="目标产品图片路径")
    parser.add_argument("--source-object", required=True, help="源产品描述")
    parser.add_argument("--target-object", required=True, help="目标产品描述")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--use-4bit", action="store_true", help="使用 4-bit 量化版本")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.template_frame).exists():
        print(f"Error: Template frame not found: {args.template_frame}")
        return 1
    if not Path(args.product_image).exists():
        print(f"Error: Product image not found: {args.product_image}")
        return 1

    # 生成目标第一帧
    generate_target_first_frame(
        template_frame_path=args.template_frame,
        product_image_path=args.product_image,
        source_object=args.source_object,
        target_object=args.target_object,
        output_path=args.output,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        image_size=(args.width, args.height),
        use_4bit=args.use_4bit,
        seed=args.seed,
    )

    print(f"\n✅ Stage 1 完成！输出: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
