#!/usr/bin/env python3
"""
Etsy 商品页面抓取工具

使用 Playwright 抓取 Etsy 商品页的图片和视频。

安装依赖:
    pip install playwright
    playwright install chromium

用法:
    python scrape_etsy.py https://www.etsy.com/listing/489997879/family-charm-necklace
    python scrape_etsy.py https://www.etsy.com/listing/489997879 --output ./downloads/
    python scrape_etsy.py urls.txt --output ./downloads/  # 批量抓取
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("请安装 playwright: pip install playwright && playwright install chromium")
    exit(1)

import requests


def extract_listing_id(url: str) -> str:
    """从 URL 提取 listing ID"""
    match = re.search(r'/listing/(\d+)', url)
    if match:
        return match.group(1)
    raise ValueError(f"无法从 URL 提取 listing ID: {url}")


def scrape_etsy_listing(url: str, headless: bool = True) -> dict:
    """
    抓取 Etsy 商品页面的图片和视频

    Returns:
        {
            "listing_id": "489997879",
            "title": "Family Charm Necklace...",
            "url": "https://...",
            "images": ["https://...jpg", ...],
            "videos": ["https://...mp4", ...],
        }
    """
    listing_id = extract_listing_id(url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        print(f"Loading {url}...")
        page.goto(url, wait_until="networkidle", timeout=60000)

        # 等待页面加载
        time.sleep(2)

        # 获取标题
        title = ""
        try:
            title_el = page.query_selector("h1")
            if title_el:
                title = title_el.inner_text().strip()
        except:
            pass

        # 方法1: 从页面 script 标签提取 JSON 数据
        images = []
        videos = []

        # 查找包含 listing 数据的 script 标签
        scripts = page.query_selector_all("script")
        for script in scripts:
            try:
                content = script.inner_html()

                # 查找图片 URL (高清版本)
                # Etsy 图片格式: https://i.etsystatic.com/xxxxx/r/il/xxxxx/xxxxx/il_1588xN.xxxxx.jpg
                img_matches = re.findall(r'https://i\.etsystatic\.com/[^"\']+?il_1588xN[^"\']+\.(?:jpg|png|webp)', content)
                images.extend(img_matches)

                # 查找视频 URL
                # Etsy 视频格式: https://v.etsystatic.com/video/upload/xxxxx.mp4
                video_matches = re.findall(r'https://v\.etsystatic\.com/video/upload/[^"\']+\.mp4', content)
                videos.extend(video_matches)

            except:
                continue

        # 方法2: 从 img 标签提取
        img_elements = page.query_selector_all("img[src*='etsystatic.com']")
        for img in img_elements:
            src = img.get_attribute("src")
            if src and "il_" in src:
                # 转换为高清版本
                high_res = re.sub(r'il_\d+x\d+', 'il_1588xN', src)
                high_res = re.sub(r'il_\d+xN', 'il_1588xN', high_res)
                if high_res not in images:
                    images.append(high_res)

        # 方法3: 从 video 标签提取
        video_elements = page.query_selector_all("video source")
        for video in video_elements:
            src = video.get_attribute("src")
            if src and "etsystatic.com" in src:
                if src not in videos:
                    videos.append(src)

        # 方法4: 查找 carousel 中的媒体
        carousel_items = page.query_selector_all("[data-carousel-item]")
        for item in carousel_items:
            # 检查是否有视频
            video_el = item.query_selector("video")
            if video_el:
                source = video_el.query_selector("source")
                if source:
                    src = source.get_attribute("src")
                    if src and src not in videos:
                        videos.append(src)

        browser.close()

        # 去重并排序
        images = list(dict.fromkeys(images))
        videos = list(dict.fromkeys(videos))

        return {
            "listing_id": listing_id,
            "title": title,
            "url": url,
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


def scrape_and_download(url: str, output_dir: Path, headless: bool = True) -> dict:
    """抓取并下载商品媒体文件"""
    data = scrape_etsy_listing(url, headless=headless)
    listing_id = data["listing_id"]

    # 创建输出目录
    listing_dir = output_dir / listing_id
    listing_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {
        "listing_id": listing_id,
        "title": data["title"],
        "url": url,
        "images": [],
        "videos": [],
    }

    # 下载图片
    print(f"\n找到 {len(data['images'])} 张图片")
    for i, img_url in enumerate(data["images"]):
        ext = Path(urlparse(img_url).path).suffix or ".jpg"
        filename = f"image_{i+1:02d}{ext}"
        filepath = listing_dir / filename

        print(f"  下载图片 {i+1}/{len(data['images'])}: {filename}")
        if download_file(img_url, filepath):
            downloaded["images"].append({
                "url": img_url,
                "path": str(filepath),
            })

    # 下载视频
    print(f"\n找到 {len(data['videos'])} 个视频")
    for i, video_url in enumerate(data["videos"]):
        ext = Path(urlparse(video_url).path).suffix or ".mp4"
        filename = f"video_{i+1:02d}{ext}"
        filepath = listing_dir / filename

        print(f"  下载视频 {i+1}/{len(data['videos'])}: {filename}")
        if download_file(video_url, filepath):
            downloaded["videos"].append({
                "url": video_url,
                "path": str(filepath),
            })

    # 保存元数据
    meta_path = listing_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, indent=2, ensure_ascii=False)
    print(f"\n元数据已保存到: {meta_path}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="抓取 Etsy 商品页面的图片和视频")
    parser.add_argument("input", help="Etsy URL 或包含 URL 的文件")
    parser.add_argument("--output", "-o", default="./etsy_downloads", help="输出目录")
    parser.add_argument("--no-download", action="store_true", help="只提取 URL，不下载")
    parser.add_argument("--visible", action="store_true", help="显示浏览器窗口（调试用）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 判断输入是 URL 还是文件
    if args.input.startswith("http"):
        urls = [args.input]
    else:
        # 从文件读取 URL 列表
        with open(args.input, "r") as f:
            urls = [line.strip() for line in f if line.strip() and line.startswith("http")]

    print(f"共 {len(urls)} 个 URL 待处理\n")

    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] 处理: {url}")

        try:
            if args.no_download:
                data = scrape_etsy_listing(url, headless=not args.visible)
                print(f"  标题: {data['title']}")
                print(f"  图片: {len(data['images'])} 张")
                print(f"  视频: {len(data['videos'])} 个")
                for img in data["images"][:3]:
                    print(f"    - {img[:80]}...")
                for vid in data["videos"]:
                    print(f"    - {vid[:80]}...")
            else:
                scrape_and_download(url, output_dir, headless=not args.visible)
        except Exception as e:
            print(f"  错误: {e}")

        print()


if __name__ == "__main__":
    main()
