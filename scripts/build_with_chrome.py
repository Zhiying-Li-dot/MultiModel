#!/usr/bin/env python3
"""
使用你的 Chrome profile 抓取 Etsy

⚠️ 运行前请关闭 Chrome 浏览器！

用法:
    python build_with_chrome.py --limit 5
"""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"

# 你的 Chrome profile 路径
CHROME_USER_DATA = Path.home() / "Library/Application Support/Google/Chrome"
CHROME_PROFILE = "Default"

CATEGORY_PREFIX = {"jewelry": "JEW", "home": "HOME", "beauty": "BEA"}

DEFAULT_URLS = [
    ("jewelry", "anklet", "https://www.etsy.com/listing/4419856766/double-chain-gold-anklet-layered-anklet"),
    ("jewelry", "necklace", "https://www.etsy.com/listing/1128997665/custom-birth-flower-necklace-flower"),
    ("jewelry", "bracelet", "https://www.etsy.com/listing/1074193270/custom-photo-projection-bracelet"),
]


def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"version": "1.0", "samples": []}


def save_metadata(metadata):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_next_id(metadata, category):
    prefix = CATEGORY_PREFIX.get(category, "UNK")
    existing = [s["id"] for s in metadata["samples"] if s["id"].startswith(prefix)]
    if not existing:
        return f"{prefix}001"
    max_num = max(int(id_[len(prefix):]) for id_ in existing)
    return f"{prefix}{max_num + 1:03d}"


def extract_listing_id(url):
    match = re.search(r'/listing/(\d+)', url)
    return match.group(1) if match else "unknown"


def scrape_page(page, url):
    listing_id = extract_listing_id(url)

    print(f"  加载页面...")
    page.goto(url, timeout=60000)
    time.sleep(5)

    content = page.content()
    print(f"  页面大小: {len(content)} bytes")

    # 检查是否被拦截
    if "Verification Required" in content or len(content) < 5000:
        print("  ⚠️ 需要验证，等待 60 秒让你手动完成...")
        time.sleep(60)
        content = page.content()

    # 提取标题
    title = ""
    try:
        el = page.query_selector("h1")
        if el:
            title = el.inner_text().strip()
    except:
        pass

    # 提取媒体
    images, videos = set(), set()

    for url_match in re.findall(r'"contentURL"\s*:\s*"([^"]+)"', content):
        url_clean = url_match.replace('\\/', '/')
        if 'v.etsystatic.com' in url_clean:
            if not url_clean.endswith('.mp4'):
                url_clean += '.mp4'
            videos.add(url_clean)
        elif 'i.etsystatic.com' in url_clean and 'il_' in url_clean:
            images.add(url_clean)

    print(f"  找到: {len(images)} 图片, {len(videos)} 视频")

    return {
        "listing_id": listing_id,
        "title": title,
        "images": sorted(images),
        "videos": sorted(videos),
    }


def download_file(url, path):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.etsy.com/"}
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"    ✓ {path.name} ({path.stat().st_size/1024/1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--urls", help="URL file")
    args = parser.parse_args()

    # 检查 Chrome 是否在运行
    result = subprocess.run(["pgrep", "-x", "Google Chrome"], capture_output=True)
    if result.returncode == 0:
        print("⚠️  请先关闭 Chrome 浏览器！")
        print("   然后重新运行此脚本。")
        return 1

    urls = DEFAULT_URLS[:args.limit]

    print(f"=== 使用 Chrome Profile 抓取 ===")
    print(f"Profile: {CHROME_USER_DATA / CHROME_PROFILE}")
    print(f"URLs: {len(urls)}\n")

    metadata = load_metadata()
    success = 0

    with sync_playwright() as p:
        # 使用你的 Chrome，带完整 profile
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(CHROME_USER_DATA),
            channel="chrome",  # 使用系统安装的 Chrome
            headless=False,
            args=[f"--profile-directory={CHROME_PROFILE}"],
            viewport={"width": 1920, "height": 1080},
        )

        page = browser.pages[0] if browser.pages else browser.new_page()

        for i, (cat, subcat, url) in enumerate(urls):
            print(f"\n[{i+1}/{len(urls)}] {url[:60]}...")

            listing_id = extract_listing_id(url)
            if any(s.get("source_listing_id") == listing_id for s in metadata["samples"]):
                print("  跳过（已存在）")
                continue

            try:
                data = scrape_page(page, url)

                if not data["videos"]:
                    print("  跳过（无视频）")
                    continue

                sample_id = get_next_id(metadata, cat)
                print(f"  样本: {sample_id}")

                # 下载
                video_dir = DATASET_ROOT / "videos" / cat
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"{sample_id}.mp4"
                download_file(data["videos"][0], video_path)

                image_path = None
                if data["images"]:
                    image_dir = DATASET_ROOT / "images" / cat
                    image_dir.mkdir(parents=True, exist_ok=True)
                    ext = ".jpg"
                    image_path = image_dir / f"{sample_id}_source{ext}"
                    download_file(data["images"][0], image_path)

                # 保存
                metadata["samples"].append({
                    "id": sample_id,
                    "category": cat,
                    "subcategory": subcat,
                    "source_listing_id": listing_id,
                    "template_video": {"path": f"videos/{cat}/{sample_id}.mp4", "source": "etsy", "source_url": url},
                    "source_product": {"image_path": f"images/{cat}/{sample_id}_source{ext}" if image_path else None, "description": data["title"]},
                    "target_product": {"image_path": None, "description": ""},
                    "prompts": {"source": "", "target": ""},
                    "difficulty": "medium",
                    "shot_type": "pure_product",
                    "added_date": datetime.now().strftime("%Y-%m-%d"),
                })
                save_metadata(metadata)
                success += 1

            except Exception as e:
                print(f"  错误: {e}")

            time.sleep(3)

        browser.close()

    print(f"\n=== 完成 ===")
    print(f"成功: {success}/{len(urls)}")
    print(f"总样本: {len(metadata['samples'])}")


if __name__ == "__main__":
    main()
