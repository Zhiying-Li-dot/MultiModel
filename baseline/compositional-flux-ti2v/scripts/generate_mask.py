#!/usr/bin/env python3
"""
使用 Grounded-SAM-2 (HuggingFace API) 生成物体 mask
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 添加 Grounded-SAM-2 路径
GSAM_PATH = "/data/xuhao/Grounded-SAM-2"
sys.path.insert(0, GSAM_PATH)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def generate_mask(
    image_path: str,
    output_path: str,
    text_prompt: str,
    threshold: float = 0.3,
):
    """
    使用 Grounded-SAM-2 生成物体 mask。

    Args:
        image_path: 输入图片路径
        output_path: 输出 mask 路径
        text_prompt: 物体描述（需要小写，以点结尾）
        threshold: 检测阈值
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAM2 模型路径
    sam2_checkpoint = f"{GSAM_PATH}/checkpoints/sam2.1_hiera_large.pt"
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 启用 bfloat16
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()

    print("Loading SAM2 model...")
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading Grounding DINO model from HuggingFace...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # 加载图片
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # 检测物体
    print(f"Detecting: {text_prompt}")
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        target_sizes=[image.size[::-1]],
    )

    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"]

    if len(boxes) == 0:
        print("Warning: No objects detected!")
        h, w = np.array(image).shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        print(f"Detected {len(boxes)} objects: {labels} with scores: {scores.tolist()}")

        # SAM2 分割
        masks, scores_sam, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        # 合并所有 mask
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        mask = combined_mask

    # 保存 mask
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"Saved mask to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成物体 mask")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="输出 mask 路径")
    parser.add_argument("--prompt", required=True, help="物体描述（如 'bracelet.'）")
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()

    generate_mask(
        image_path=args.input,
        output_path=args.output,
        text_prompt=args.prompt,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
