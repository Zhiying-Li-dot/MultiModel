#!/usr/bin/env python3
"""
迁移现有样本到 PVTT Benchmark 数据集结构
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
DATASET_ROOT = PROJECT_ROOT / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"

# 现有样本映射
EXISTING_SAMPLES = [
    {
        "id": "JEW001",
        "category": "jewelry",
        "subcategory": "necklace",
        "video": "family-charm-necklace-custom-birthstone.mp4",
        "image": "family-charm-necklace-custom-birthstone.webp",
        "source_url": "https://www.etsy.com/listing/489997879/family-charm-necklace-custom-birthstone",
        "title": "Family Charm Necklace, Custom Birthstone Initial Necklace",
        "source_prompt": "A delicate gold family charm necklace with birthstone pendants and leaf charms on a white gift box.",
        "target_prompt": "A silver tennis bracelet with sparkling crystals on a white gift box.",
    },
    {
        "id": "JEW002",
        "category": "jewelry",
        "subcategory": "bracelet",
        "video": "personalized-couple-bracelets-settwo.mp4",
        "image": "personalized-couple-bracelets-settwo.webp",
        "source_url": "https://www.etsy.com/listing/615384498/personalized-couple-bracelets-set-two",
        "title": "Personalized Couple Bracelets Set Two",
        "source_prompt": "Two personalized couple bracelets, one silver and one black, placed on a purple silk fabric with decorative stones.",
        "target_prompt": "A gold charm necklace with colorful gemstone pendants placed on a purple silk fabric with decorative stones.",
    },
    {
        "id": "HOME001",
        "category": "home",
        "subcategory": "pillow",
        "video": "winter-ski-gear-pillow-cover-12x20-inch.mp4",
        "image": "winter-ski-gear-pillow-cover-12x20-inch.webp",
        "source_url": "https://www.etsy.com/listing/1138633008/winter-ski-gear-pillow-cover-12x20-inch",
        "title": "Winter Ski Gear Pillow Cover 12x20 Inch",
        "source_prompt": "A decorative pillow cover with winter ski gear pattern, embroidered design featuring skiing figures on white fabric.",
        "target_prompt": "A decorative pillow cover with tropical beach pattern, embroidered design featuring palm trees and sunset on white fabric.",
    },
]


def get_video_info(video_path: Path) -> dict:
    """获取视频信息"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
        if not video_stream:
            return {}

        fps_str = video_stream.get("r_frame_rate", "24/1")
        fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        duration = float(data["format"].get("duration", 0))

        return {
            "duration_sec": round(duration, 2),
            "resolution": f"{video_stream['width']}x{video_stream['height']}",
            "fps": round(fps, 2),
            "total_frames": int(video_stream.get("nb_frames", int(duration * fps))),
        }
    except Exception as e:
        print(f"  警告: 无法获取视频信息: {e}")
        return {}


def migrate():
    print("=== 迁移现有样本到 PVTT Benchmark ===\n")

    # 创建目录
    for cat in ["jewelry", "home"]:
        (DATASET_ROOT / "videos" / cat).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "images" / cat).mkdir(parents=True, exist_ok=True)

    # 加载现有 metadata
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "version": "1.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "description": "PVTT Benchmark Dataset",
            "samples": []
        }

    existing_ids = {s["id"] for s in metadata["samples"]}

    for sample in EXISTING_SAMPLES:
        sample_id = sample["id"]

        if sample_id in existing_ids:
            print(f"跳过 {sample_id}（已存在）")
            continue

        print(f"迁移 {sample_id}: {sample['title'][:50]}...")

        # 源文件路径
        src_video = SAMPLES_DIR / sample["video"]
        src_image = SAMPLES_DIR / sample["image"]

        if not src_video.exists():
            print(f"  错误: 视频不存在 {src_video}")
            continue

        # 目标路径
        category = sample["category"]
        dst_video = DATASET_ROOT / "videos" / category / f"{sample_id}.mp4"
        dst_image = DATASET_ROOT / "images" / category / f"{sample_id}_source{src_image.suffix}"

        # 复制文件
        shutil.copy2(src_video, dst_video)
        print(f"  复制视频: {dst_video.name}")

        if src_image.exists():
            shutil.copy2(src_image, dst_image)
            print(f"  复制图片: {dst_image.name}")

        # 获取视频信息
        video_info = get_video_info(dst_video)

        # 构建样本条目
        entry = {
            "id": sample_id,
            "category": category,
            "subcategory": sample["subcategory"],
            "template_video": {
                "path": f"videos/{category}/{sample_id}.mp4",
                **video_info,
                "source": "etsy",
                "source_url": sample["source_url"],
            },
            "source_product": {
                "image_path": f"images/{category}/{sample_id}_source{src_image.suffix}" if src_image.exists() else None,
                "description": sample["title"],
            },
            "target_product": {
                "image_path": None,
                "description": "",
            },
            "prompts": {
                "source": sample["source_prompt"],
                "target": sample["target_prompt"],
            },
            "difficulty": "medium",
            "shot_type": "pure_product",
            "notes": "Migrated from data/samples/",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
        }

        metadata["samples"].append(entry)
        print(f"  添加到 metadata")

    # 保存 metadata
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n=== 完成 ===")
    print(f"总样本数: {len(metadata['samples'])}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    migrate()
