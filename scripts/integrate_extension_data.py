#!/usr/bin/env python3
"""
整合 Chrome Extension 下载的数据到 PVTT Benchmark 数据集

用法:
    python integrate_extension_data.py ~/Downloads/PVTT/
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"


def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"version": "1.0", "samples": []}


def save_metadata(metadata):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_video_info(video_path):
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", "-show_streams", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        vs = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
        if vs:
            fps = eval(vs.get("r_frame_rate", "24/1"))
            dur = float(data["format"].get("duration", 0))
            return {
                "duration_sec": round(dur, 2),
                "resolution": f"{vs['width']}x{vs['height']}",
                "fps": round(fps, 2),
                "total_frames": int(dur * fps),
            }
    except:
        pass
    return {}


def integrate(source_dir):
    source_path = Path(source_dir)

    # 查找 extension metadata
    ext_metadata_file = source_path / "PVTT_metadata.json"
    if not ext_metadata_file.exists():
        print(f"❌ 未找到 {ext_metadata_file}")
        print("   请确保已用 Chrome Extension 下载过数据")
        return 1

    # 加载两边的 metadata
    with open(ext_metadata_file) as f:
        ext_metadata = json.load(f)

    dataset_metadata = load_metadata()

    existing_ids = {s.get("source_listing_id") for s in dataset_metadata["samples"]}

    print(f"=== 整合 Chrome Extension 数据 ===")
    print(f"源目录: {source_path}")
    print(f"Extension 样本: {len(ext_metadata['samples'])}")
    print(f"数据集已有: {len(dataset_metadata['samples'])}\n")

    integrated = 0

    for sample in ext_metadata["samples"]:
        sample_id = sample["id"]
        listing_id = sample.get("source_listing_id")

        # 检查重复
        if listing_id in existing_ids:
            print(f"跳过 {sample_id}（listing {listing_id} 已存在）")
            continue

        print(f"整合 {sample_id}: {sample['source_product']['description'][:50]}...")

        category = sample["category"]

        # 复制视频
        src_video = source_path / sample["template_video"]["filename"]
        if not src_video.exists():
            print(f"  ⚠️  视频不存在: {src_video}")
            continue

        dst_video_dir = DATASET_ROOT / "videos" / category
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / f"{sample_id}.mp4"

        shutil.copy2(src_video, dst_video)
        print(f"  ✓ 视频: {dst_video.name}")

        # 复制图片
        dst_image = None
        if sample["source_product"]["filename"]:
            src_image = source_path / sample["source_product"]["filename"]
            if src_image.exists():
                ext = src_image.suffix
                dst_image_dir = DATASET_ROOT / "images" / category
                dst_image_dir.mkdir(parents=True, exist_ok=True)
                dst_image = dst_image_dir / f"{sample_id}_source{ext}"
                shutil.copy2(src_image, dst_image)
                print(f"  ✓ 图片: {dst_image.name}")

        # 获取视频信息
        video_info = get_video_info(dst_video)

        # 构建新的 metadata 条目
        new_sample = {
            "id": sample_id,
            "category": sample["category"],
            "subcategory": sample["subcategory"],
            "source_listing_id": listing_id,
            "template_video": {
                "path": f"videos/{category}/{sample_id}.mp4",
                **video_info,
                "source": "etsy",
                "source_url": sample["template_video"]["source_url"],
            },
            "source_product": {
                "image_path": f"images/{category}/{sample_id}_source{ext}" if dst_image else None,
                "description": sample["source_product"]["description"],
            },
            "target_product": {
                "image_path": None,
                "description": "",
            },
            "prompts": {
                "source": sample["prompts"]["source"],
                "target": sample["prompts"]["target"],
            },
            "difficulty": sample["difficulty"],
            "shot_type": sample["shot_type"],
            "notes": "Collected via Chrome Extension",
            "added_date": sample["added_date"],
        }

        dataset_metadata["samples"].append(new_sample)
        integrated += 1

    # 保存更新后的 metadata
    save_metadata(dataset_metadata)

    print(f"\n=== 完成 ===")
    print(f"新增样本: {integrated}")
    print(f"总样本数: {len(dataset_metadata['samples'])}")
    print(f"Metadata: {METADATA_FILE}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="整合 Chrome Extension 数据")
    parser.add_argument("source_dir", help="Extension 下载目录（默认 ~/Downloads/PVTT/）")
    args = parser.parse_args()

    return integrate(args.source_dir)


if __name__ == "__main__":
    exit(main())
