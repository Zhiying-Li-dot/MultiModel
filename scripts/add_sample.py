#!/usr/bin/env python3
"""
添加样本到 PVTT 数据集

用法:
    python add_sample.py \
        --video video.mp4 \
        --source-image source.jpg \
        --target-image target.jpg \
        --category jewelry \
        --source-prompt "Two bracelets on silk..." \
        --target-prompt "A gold necklace on silk..."

交互模式:
    python add_sample.py --interactive
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# 数据集根目录
DATASET_ROOT = Path(__file__).parent.parent / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"

CATEGORY_PREFIX = {
    "jewelry": "JEW",
    "home": "HOME",
    "beauty": "BEA",
    "fashion": "FASH",
    "electronics": "ELEC",
}

SHOT_TYPES = ["pure_product", "product_closeup", "interaction", "wearing"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard", "expert"]


def load_metadata() -> dict:
    """加载现有 metadata"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": "1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "samples": []
    }


def save_metadata(metadata: dict):
    """保存 metadata"""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_next_id(metadata: dict, category: str) -> str:
    """获取下一个可用 ID"""
    prefix = CATEGORY_PREFIX.get(category, "UNK")
    existing = [s["id"] for s in metadata["samples"] if s["id"].startswith(prefix)]

    if not existing:
        return f"{prefix}001"

    # 找最大编号
    max_num = 0
    for id_ in existing:
        try:
            num = int(id_[len(prefix):])
            max_num = max(max_num, num)
        except ValueError:
            pass

    return f"{prefix}{max_num + 1:03d}"


def extract_video_info(video_path: str) -> dict:
    """提取视频信息"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError("No video stream found")

    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    duration = float(data["format"].get("duration", 0))
    total_frames = int(video_stream.get("nb_frames", int(duration * fps)))

    return {
        "duration_sec": round(duration, 2),
        "resolution": f"{video_stream['width']}x{video_stream['height']}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
    }


def add_sample(
    video_path: str,
    source_image: str,
    target_image: str,
    category: str,
    subcategory: str,
    source_prompt: str,
    target_prompt: str,
    source_description: str,
    target_description: str,
    shot_type: str = "pure_product",
    difficulty: str = "medium",
    source_url: str = "",
    notes: str = "",
) -> str:
    """添加样本到数据集

    Returns:
        新样本的 ID
    """
    # 加载现有数据
    metadata = load_metadata()

    # 生成 ID
    sample_id = get_next_id(metadata, category)

    # 创建目标路径
    video_dst = DATASET_ROOT / "videos" / category / f"{sample_id}.mp4"
    source_img_dst = DATASET_ROOT / "images" / category / f"{sample_id}_source.jpg"
    target_img_dst = DATASET_ROOT / "images" / category / f"{sample_id}_target.jpg"

    # 复制文件
    video_dst.parent.mkdir(parents=True, exist_ok=True)
    source_img_dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(video_path, video_dst)
    shutil.copy2(source_image, source_img_dst)
    shutil.copy2(target_image, target_img_dst)

    # 提取视频信息
    video_info = extract_video_info(str(video_dst))

    # 构建样本条目
    sample = {
        "id": sample_id,
        "category": category,
        "subcategory": subcategory,
        "template_video": {
            "path": f"videos/{category}/{sample_id}.mp4",
            **video_info,
            "source": "etsy" if "etsy" in source_url.lower() else "unknown",
            "source_url": source_url,
        },
        "source_product": {
            "image_path": f"images/{category}/{sample_id}_source.jpg",
            "description": source_description,
        },
        "target_product": {
            "image_path": f"images/{category}/{sample_id}_target.jpg",
            "description": target_description,
        },
        "prompts": {
            "source": source_prompt,
            "target": target_prompt,
        },
        "difficulty": difficulty,
        "shot_type": shot_type,
        "notes": notes,
        "added_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # 添加到 metadata
    metadata["samples"].append(sample)
    save_metadata(metadata)

    print(f"Added sample: {sample_id}")
    print(f"  Video: {video_dst}")
    print(f"  Source image: {source_img_dst}")
    print(f"  Target image: {target_img_dst}")

    return sample_id


def interactive_add():
    """交互式添加样本"""
    print("=== PVTT 数据集样本添加 ===\n")

    # 基本信息
    video_path = input("视频文件路径: ").strip()
    source_image = input("源产品图片路径: ").strip()
    target_image = input("目标产品图片路径: ").strip()

    # 品类
    print(f"\n可选品类: {', '.join(CATEGORY_PREFIX.keys())}")
    category = input("品类: ").strip().lower()
    subcategory = input("子品类 (如 bracelet, necklace): ").strip().lower()

    # 描述
    source_description = input("\n源产品描述 (简短): ").strip()
    target_description = input("目标产品描述 (简短): ").strip()

    # Prompts
    print("\n(Prompts 应包含产品和背景描述)")
    source_prompt = input("Source Prompt: ").strip()
    target_prompt = input("Target Prompt: ").strip()

    # 类型和难度
    print(f"\n镜头类型: {', '.join(SHOT_TYPES)}")
    shot_type = input("镜头类型 [pure_product]: ").strip() or "pure_product"

    print(f"难度: {', '.join(DIFFICULTY_LEVELS)}")
    difficulty = input("难度 [medium]: ").strip() or "medium"

    # 可选信息
    source_url = input("\n来源 URL (可选): ").strip()
    notes = input("备注 (可选): ").strip()

    # 确认
    print("\n=== 确认信息 ===")
    print(f"视频: {video_path}")
    print(f"品类: {category}/{subcategory}")
    print(f"类型: {shot_type}, 难度: {difficulty}")

    confirm = input("\n确认添加? [y/N]: ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    # 添加
    sample_id = add_sample(
        video_path=video_path,
        source_image=source_image,
        target_image=target_image,
        category=category,
        subcategory=subcategory,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        source_description=source_description,
        target_description=target_description,
        shot_type=shot_type,
        difficulty=difficulty,
        source_url=source_url,
        notes=notes,
    )

    print(f"\n成功添加样本: {sample_id}")


def main():
    parser = argparse.ArgumentParser(description="添加样本到 PVTT 数据集")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")

    # 非交互模式参数
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--source-image", help="源产品图片路径")
    parser.add_argument("--target-image", help="目标产品图片路径")
    parser.add_argument("--category", help="品类")
    parser.add_argument("--subcategory", default="", help="子品类")
    parser.add_argument("--source-prompt", help="Source prompt")
    parser.add_argument("--target-prompt", help="Target prompt")
    parser.add_argument("--source-desc", default="", help="源产品描述")
    parser.add_argument("--target-desc", default="", help="目标产品描述")
    parser.add_argument("--shot-type", default="pure_product", help="镜头类型")
    parser.add_argument("--difficulty", default="medium", help="难度")
    parser.add_argument("--source-url", default="", help="来源 URL")
    parser.add_argument("--notes", default="", help="备注")

    args = parser.parse_args()

    if args.interactive:
        interactive_add()
    elif args.video:
        add_sample(
            video_path=args.video,
            source_image=args.source_image,
            target_image=args.target_image,
            category=args.category,
            subcategory=args.subcategory,
            source_prompt=args.source_prompt,
            target_prompt=args.target_prompt,
            source_description=args.source_desc,
            target_description=args.target_desc,
            shot_type=args.shot_type,
            difficulty=args.difficulty,
            source_url=args.source_url,
            notes=args.notes,
        )
    else:
        parser.print_help()
        print("\n使用 --interactive 进入交互模式")


if __name__ == "__main__":
    main()
