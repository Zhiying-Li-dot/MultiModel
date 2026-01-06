#!/usr/bin/env python3
"""
手动收集的 Etsy 视频下载工具

使用方法：
1. 在浏览器打开 Etsy 商品页
2. F12 → Network → 筛选 mp4
3. 播放视频，复制视频 URL
4. 运行此脚本

用法:
    python download_manual.py

或直接下载单个 URL:
    python download_manual.py "https://v.etsystatic.com/video/..." --id JEW003 --category jewelry
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"

CATEGORY_PREFIX = {
    "jewelry": "JEW",
    "home": "HOME",
    "beauty": "BEA",
    "fashion": "FASH",
    "electronics": "ELEC",
}


def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"version": "1.0", "samples": []}


def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_next_id(metadata, category):
    prefix = CATEGORY_PREFIX.get(category, "UNK")
    existing = [s["id"] for s in metadata["samples"] if s["id"].startswith(prefix)]
    if not existing:
        return f"{prefix}001"
    max_num = max(int(id_[len(prefix):]) for id_ in existing)
    return f"{prefix}{max_num + 1:03d}"


def download_file(url, output_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer": "https://www.etsy.com/",
    }
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  下载完成: {output_path.name} ({size_mb:.1f} MB)")
    return True


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
                "total_frames": int(vs.get("nb_frames", int(dur * fps))),
            }
    except:
        pass
    return {}


def add_sample(video_url, image_url=None, category="jewelry", subcategory="",
               title="", source_url="", source_prompt="", target_prompt=""):
    metadata = load_metadata()
    sample_id = get_next_id(metadata, category)

    print(f"\n添加样本: {sample_id}")

    # 创建目录
    video_dir = DATASET_ROOT / "videos" / category
    image_dir = DATASET_ROOT / "images" / category
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 下载视频
    video_path = video_dir / f"{sample_id}.mp4"
    print(f"  下载视频: {video_url[:60]}...")
    download_file(video_url, video_path)

    # 下载图片（如果有）
    image_path = None
    if image_url:
        ext = ".jpg" if ".jpg" in image_url else ".webp"
        image_path = image_dir / f"{sample_id}_source{ext}"
        print(f"  下载图片: {image_url[:60]}...")
        download_file(image_url, image_path)

    # 获取视频信息
    video_info = get_video_info(video_path)

    # 添加到 metadata
    sample = {
        "id": sample_id,
        "category": category,
        "subcategory": subcategory,
        "template_video": {
            "path": f"videos/{category}/{sample_id}.mp4",
            **video_info,
            "source": "etsy",
            "source_url": source_url,
        },
        "source_product": {
            "image_path": f"images/{category}/{sample_id}_source{ext}" if image_path else None,
            "description": title,
        },
        "target_product": {"image_path": None, "description": ""},
        "prompts": {"source": source_prompt, "target": target_prompt},
        "difficulty": "medium",
        "shot_type": "pure_product",
        "notes": f"Manual collection on {datetime.now().strftime('%Y-%m-%d')}",
        "added_date": datetime.now().strftime("%Y-%m-%d"),
    }

    metadata["samples"].append(sample)
    save_metadata(metadata)

    print(f"\n✓ 已添加 {sample_id}")
    print(f"  总样本数: {len(metadata['samples'])}")
    return sample_id


def interactive_add():
    print("=== PVTT 手动添加样本 ===\n")

    video_url = input("视频 URL (v.etsystatic.com/...): ").strip()
    if not video_url:
        print("取消")
        return

    # 确保是完整 URL 且有 .mp4
    if not video_url.startswith("http"):
        video_url = "https://" + video_url
    if not video_url.endswith(".mp4"):
        video_url += ".mp4"

    image_url = input("图片 URL (可选，直接回车跳过): ").strip()
    if image_url and not image_url.startswith("http"):
        image_url = "https://" + image_url

    print("\n品类: jewelry, home, beauty, fashion, electronics")
    category = input("品类 [jewelry]: ").strip() or "jewelry"
    subcategory = input("子品类 (如 necklace, bracelet): ").strip()

    title = input("商品标题: ").strip()
    source_url = input("Etsy 页面 URL (可选): ").strip()

    print("\n(Prompt 描述产品和背景，用于编辑实验)")
    source_prompt = input("Source Prompt (可选): ").strip()
    target_prompt = input("Target Prompt (可选): ").strip()

    add_sample(
        video_url=video_url,
        image_url=image_url or None,
        category=category,
        subcategory=subcategory,
        title=title,
        source_url=source_url,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )


def main():
    parser = argparse.ArgumentParser(description="手动添加 PVTT 样本")
    parser.add_argument("video_url", nargs="?", help="视频 URL")
    parser.add_argument("--image", help="图片 URL")
    parser.add_argument("--category", default="jewelry", help="品类")
    parser.add_argument("--subcategory", default="", help="子品类")
    parser.add_argument("--title", default="", help="标题")
    parser.add_argument("--source-url", default="", help="来源 URL")
    args = parser.parse_args()

    if args.video_url:
        add_sample(
            video_url=args.video_url,
            image_url=args.image,
            category=args.category,
            subcategory=args.subcategory,
            title=args.title,
            source_url=args.source_url,
        )
    else:
        interactive_add()


if __name__ == "__main__":
    main()
