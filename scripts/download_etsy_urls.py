#!/usr/bin/env python3
"""
下载 Etsy 媒体文件

配合 etsy_bookmarklet.js 使用，下载提取的 URL。

用法:
    # 从 JSON 文件下载
    python download_etsy_urls.py urls.json --output ./downloads/

    # 从剪贴板下载 (macOS)
    pbpaste | python download_etsy_urls.py - --output ./downloads/

    # 直接下载单个 URL
    python download_etsy_urls.py "https://v.etsystatic.com/video/..." --output ./downloads/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


def download_file(url: str, output_path: Path) -> bool:
    """下载单个文件"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://www.etsy.com/",
        }
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"\r  下载中: {progress:.1f}%", end="", flush=True)

        print(f"\r  完成: {output_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  失败: {e}")
        return False


def parse_input(input_data: str) -> dict:
    """解析输入数据，支持 JSON 或纯文本 URL 列表"""

    # 尝试解析为 JSON
    try:
        data = json.loads(input_data)
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # URL 列表
            images = [u for u in data if any(ext in u for ext in ['.jpg', '.png', '.webp'])]
            videos = [u for u in data if '.mp4' in u]
            return {"images": images, "videos": videos}
    except json.JSONDecodeError:
        pass

    # 解析为纯文本 URL 列表
    urls = re.findall(r'https?://[^\s"\'<>]+', input_data)
    images = [u for u in urls if any(ext in u for ext in ['.jpg', '.png', '.webp'])]
    videos = [u for u in urls if '.mp4' in u]

    # 尝试提取 listing_id
    listing_id = "unknown"
    id_match = re.search(r'listing[/_](\d+)', input_data)
    if id_match:
        listing_id = id_match.group(1)

    return {
        "listing_id": listing_id,
        "images": images,
        "videos": videos,
    }


def main():
    parser = argparse.ArgumentParser(description="下载 Etsy 媒体文件")
    parser.add_argument("input", help="JSON 文件、URL、或 - 表示从 stdin 读取")
    parser.add_argument("--output", "-o", default="./etsy_downloads", help="输出目录")
    parser.add_argument("--images-only", action="store_true", help="只下载图片")
    parser.add_argument("--videos-only", action="store_true", help="只下载视频")
    args = parser.parse_args()

    # 读取输入
    if args.input == "-":
        input_data = sys.stdin.read()
    elif args.input.startswith("http"):
        # 单个 URL
        input_data = args.input
    else:
        with open(args.input, "r") as f:
            input_data = f.read()

    data = parse_input(input_data)

    listing_id = data.get("listing_id", "unknown")
    images = data.get("images", [])
    videos = data.get("videos", [])

    print(f"Listing ID: {listing_id}")
    print(f"图片: {len(images)} 个")
    print(f"视频: {len(videos)} 个")

    if not images and not videos:
        print("\n未找到有效的媒体 URL")
        return 1

    # 创建输出目录
    output_dir = Path(args.output) / listing_id
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {"listing_id": listing_id, "images": [], "videos": []}

    # 下载图片
    if not args.videos_only and images:
        print(f"\n下载图片到 {output_dir}/")
        for i, url in enumerate(images):
            ext = Path(urlparse(url).path).suffix or ".jpg"
            filename = f"image_{i+1:02d}{ext}"
            filepath = output_dir / filename

            if download_file(url, filepath):
                downloaded["images"].append({"url": url, "path": str(filepath)})

    # 下载视频
    if not args.images_only and videos:
        print(f"\n下载视频到 {output_dir}/")
        for i, url in enumerate(videos):
            ext = Path(urlparse(url).path).suffix or ".mp4"
            filename = f"video_{i+1:02d}{ext}"
            filepath = output_dir / filename

            if download_file(url, filepath):
                downloaded["videos"].append({"url": url, "path": str(filepath)})

    # 保存元数据
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**data, "downloaded": downloaded}, f, indent=2, ensure_ascii=False)

    print(f"\n完成！元数据已保存到: {meta_path}")
    print(f"下载了 {len(downloaded['images'])} 张图片，{len(downloaded['videos'])} 个视频")

    return 0


if __name__ == "__main__":
    exit(main())
