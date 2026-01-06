#!/usr/bin/env python3
"""
解析保存的 Etsy HTML 页面

当 Playwright 被拦截时，可以手动在浏览器中保存 HTML：
1. 打开 Etsy 商品页
2. 右键 → 查看页面源代码 (View Page Source)
3. Cmd+S 保存为 .html 文件
4. 用本脚本解析

用法:
    python parse_etsy_html.py saved_page.html
    python parse_etsy_html.py saved_page.html --download --output ./downloads/
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


def parse_etsy_html(html_path: str) -> dict:
    """解析 Etsy HTML 页面，提取图片和视频 URL"""

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取 listing ID
    listing_id = "unknown"
    id_match = re.search(r'/listing/(\d+)', content)
    if id_match:
        listing_id = id_match.group(1)

    # 提取标题
    title = ""
    title_match = re.search(r'<title>([^<]+)</title>', content)
    if title_match:
        title = title_match.group(1).split(" - Etsy")[0].strip()

    # 提取图片 URL
    images = []

    # 高清图片模式
    img_patterns = [
        r'https://i\.etsystatic\.com/\d+/r/il/[a-f0-9]+/\d+/il_1588xN\.[a-f0-9]+\.(?:jpg|png|webp)',
        r'https://i\.etsystatic\.com/\d+/r/il/[a-f0-9]+/\d+/il_fullxfull\.[a-f0-9]+\.(?:jpg|png|webp)',
        r'https://i\.etsystatic\.com/[^"\'>\s]+\.(?:jpg|png|webp)',
    ]

    for pattern in img_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # 转换为最高清版本
            high_res = re.sub(r'il_\d+x\d+', 'il_1588xN', match)
            high_res = re.sub(r'il_\d+xN', 'il_1588xN', high_res)
            if high_res not in images:
                images.append(high_res)

    # 提取视频 URL
    videos = []

    video_patterns = [
        r'https://v\.etsystatic\.com/video/upload/[^"\'>\s]+\.mp4',
        r'https://[^"\'>\s]*etsystatic\.com[^"\'>\s]*\.mp4',
    ]

    for pattern in video_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if match not in videos:
                videos.append(match)

    # 尝试从 JSON 数据中提取
    # Etsy 页面通常包含 __INITIAL_STATE__ 或类似的 JSON 数据
    json_patterns = [
        r'window\.__INITIAL_STATE__\s*=\s*({.+?});',
        r'"listing":\s*({.+?})\s*[,}]',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                # 在 JSON 中查找媒体 URL
                media_urls = re.findall(r'https://[iv]\.etsystatic\.com/[^"]+', match)
                for url in media_urls:
                    if url.endswith(('.jpg', '.png', '.webp')):
                        high_res = re.sub(r'il_\d+x\d+', 'il_1588xN', url)
                        if high_res not in images:
                            images.append(high_res)
                    elif url.endswith('.mp4'):
                        if url not in videos:
                            videos.append(url)
            except:
                continue

    # 去重保持顺序
    images = list(dict.fromkeys(images))
    videos = list(dict.fromkeys(videos))

    # 过滤掉缩略图和 icon
    images = [img for img in images if "il_" in img and "icon" not in img.lower()]

    return {
        "listing_id": listing_id,
        "title": title,
        "source_file": html_path,
        "images": images,
        "videos": videos,
    }


def download_file(url: str, output_path: Path) -> bool:
    """下载文件"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://www.etsy.com/",
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="解析保存的 Etsy HTML 页面")
    parser.add_argument("html_file", help="HTML 文件路径")
    parser.add_argument("--download", "-d", action="store_true", help="下载媒体文件")
    parser.add_argument("--output", "-o", default="./etsy_downloads", help="输出目录")
    args = parser.parse_args()

    print(f"解析: {args.html_file}\n")
    data = parse_etsy_html(args.html_file)

    print(f"Listing ID: {data['listing_id']}")
    print(f"标题: {data['title']}")
    print(f"\n图片 ({len(data['images'])} 张):")
    for i, img in enumerate(data["images"][:10]):  # 只显示前10张
        print(f"  {i+1}. {img}")
    if len(data["images"]) > 10:
        print(f"  ... 还有 {len(data['images']) - 10} 张")

    print(f"\n视频 ({len(data['videos'])} 个):")
    for i, vid in enumerate(data["videos"]):
        print(f"  {i+1}. {vid}")

    if args.download:
        output_dir = Path(args.output) / data["listing_id"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n下载到: {output_dir}")

        # 下载图片
        for i, img_url in enumerate(data["images"]):
            ext = Path(urlparse(img_url).path).suffix or ".jpg"
            filename = f"image_{i+1:02d}{ext}"
            filepath = output_dir / filename
            print(f"  下载图片 {i+1}: {filename}")
            download_file(img_url, filepath)

        # 下载视频
        for i, video_url in enumerate(data["videos"]):
            ext = Path(urlparse(video_url).path).suffix or ".mp4"
            filename = f"video_{i+1:02d}{ext}"
            filepath = output_dir / filename
            print(f"  下载视频 {i+1}: {filename}")
            download_file(video_url, filepath)

        # 保存元数据
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n元数据已保存: {meta_path}")

    else:
        # 输出 JSON
        print(f"\nJSON 输出:")
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
