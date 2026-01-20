#!/usr/bin/env python3
"""
LaMa Inpainting - 基于纹理的物体移除

使用 LaMa (Large Mask Inpainting) 根据周围纹理填充 mask 区域。
适合物体移除场景（不需要 prompt）。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama


def lama_inpaint(
    input_image_path: str,
    mask_path: str,
    output_path: str,
    dilate_kernel: int = 0,
    mask_mode: str = "original",
):
    """
    使用 LaMa 进行 inpainting。

    Args:
        input_image_path: 输入图片路径
        mask_path: mask 图片路径（白色=要修复的区域）
        output_path: 输出图片路径
        dilate_kernel: 膨胀 kernel 大小（0=不膨胀）
        mask_mode: mask 模式 (original/bbox/convex)
    """
    print(f"Loading image: {input_image_path}")
    image = Image.open(input_image_path).convert("RGB")

    print(f"Loading mask: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 膨胀处理
    if dilate_kernel > 0:
        print(f"Dilating mask with kernel size {dilate_kernel}")
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Mask 形状变换
    if mask_mode == "bbox":
        print("Converting mask to bounding box")
        coords = np.where(mask > 127)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox_mask = np.zeros_like(mask)
            bbox_mask[y_min:y_max+1, x_min:x_max+1] = 255
            mask = bbox_mask
    elif mask_mode == "convex":
        print("Converting mask to convex hull")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            convex_mask = np.zeros_like(mask)
            cv2.fillPoly(convex_mask, [hull], 255)
            mask = convex_mask

    # 转换为 PIL Image
    mask_pil = Image.fromarray(mask).convert("L")

    # 确保 mask 和图片尺寸一致
    if mask_pil.size != image.size:
        print(f"Resizing mask from {mask_pil.size} to {image.size}")
        mask_pil = mask_pil.resize(image.size, Image.NEAREST)

    # 运行 LaMa
    print("Running LaMa inpainting...")
    simple_lama = SimpleLama()
    result = simple_lama(image, mask_pil)

    # 保存结果
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="LaMa Inpainting 物体移除")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--mask", required=True, help="mask 图片路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument("--dilate", type=int, default=0, help="膨胀 kernel 大小（0=不膨胀）")
    parser.add_argument("--mask-mode", choices=["original", "bbox", "convex"], default="original",
                        help="mask 模式: original/bbox/convex")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input image not found: {args.input}")
        return 1

    if not Path(args.mask).exists():
        print(f"Error: Mask image not found: {args.mask}")
        return 1

    lama_inpaint(
        input_image_path=args.input,
        mask_path=args.mask,
        output_path=args.output,
        dilate_kernel=args.dilate,
        mask_mode=args.mask_mode,
    )

    return 0


if __name__ == "__main__":
    exit(main())
