#!/usr/bin/env python3
"""
PVTT 数据集构建 - 使用持久化浏览器会话

第一次运行时会打开浏览器让你手动过验证码，之后会复用 session。

用法:
    python build_dataset_persistent.py --urls urls.txt
    python build_dataset_persistent.py --limit 10
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
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# 路径配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"
# 持久化浏览器数据目录
USER_DATA_DIR = PROJECT_ROOT / ".browser_data"

CATEGORY_PREFIX = {
    "jewelry": "JEW",
    "home": "HOME",
    "beauty": "BEA",
}

# 默认 URL 列表
DEFAULT_URLS = [
    ("jewelry", "necklace", "https://www.etsy.com/listing/489997879/family-charm-necklace-custom-birthstone"),
    ("jewelry", "bracelet", "https://www.etsy.com/listing/615384498/personalized-couple-bracelets-set-two"),
    ("jewelry", "anklet", "https://www.etsy.com/listing/4419856766/double-chain-gold-anklet-layered-anklet"),
]


def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"version": "1.0", "created": datetime.now().strftime("%Y-%m-%d"), "samples": []}


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


def wait_for_human_verification(page, timeout=120):
    """等待用户手动完成验证，自动检测验证完成"""
    print("\n" + "="*50)
    print("⚠️  检测到验证页面！")
    print("请在浏览器中完成验证（滑动拼图等）")
    print(f"等待验证完成（最多 {timeout} 秒）...")
    print("="*50 + "\n")

    # 等待页面变化（验证完成后会跳转到商品页）
    start = time.time()
    while time.time() - start < timeout:
        content = page.content()
        # 验证完成的标志：页面有 h1 且没有 verification
        if "Verification" not in content and len(content) > 10000:
            print("  ✓ 验证完成！")
            time.sleep(2)
            return True
        time.sleep(2)

    print("  ✗ 验证超时")
    return False


def scrape_etsy_page(page, url):
    """抓取 Etsy 页面"""
    listing_id = extract_listing_id(url)

    print(f"  加载页面...")
    page.goto(url, timeout=60000)
    time.sleep(3)

    # 检查是否需要验证
    content = page.content()
    if "Verification Required" in content or "captcha" in content.lower() or len(content) < 5000:
        if not wait_for_human_verification(page):
            return {"listing_id": listing_id, "title": "", "url": url, "images": [], "videos": []}
        content = page.content()

    # 等待页面加载
    try:
        page.wait_for_selector("h1", timeout=10000)
    except:
        pass

    time.sleep(2)

    # 获取标题
    title = ""
    try:
        title_el = page.query_selector("h1")
        if title_el:
            title = title_el.inner_text().strip()
    except:
        pass

    # 提取媒体
    images = set()
    videos = set()

    content = page.content()
    print(f"  页面大小: {len(content)} bytes")

    # 从 JSON-LD 提取
    content_urls = re.findall(r'"contentURL"\s*:\s*"([^"]+)"', content)
    for url_escaped in content_urls:
        url_clean = url_escaped.replace('\\/', '/')
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
        "url": url,
        "images": sorted(list(images)),
        "videos": sorted(list(videos)),
    }


def download_file(url, output_path):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.etsy.com/"}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"    ✓ {output_path.name} ({size_mb:.1f} MB)")
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
            return {"duration_sec": round(dur, 2), "resolution": f"{vs['width']}x{vs['height']}",
                    "fps": round(fps, 2), "total_frames": int(dur * fps)}
    except:
        pass
    return {}


def process_listing(page, url, category, subcategory, metadata):
    listing_id = extract_listing_id(url)

    # 检查重复
    if any(s.get("source_listing_id") == listing_id for s in metadata["samples"]):
        print(f"  跳过（已存在）")
        return False

    data = scrape_etsy_page(page, url)

    if not data["videos"]:
        print(f"  跳过（无视频）")
        return False

    sample_id = get_next_id(metadata, category)
    print(f"  样本: {sample_id} - {data['title'][:50]}")

    # 创建目录
    video_dir = DATASET_ROOT / "videos" / category
    image_dir = DATASET_ROOT / "images" / category
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 下载
    video_path = video_dir / f"{sample_id}.mp4"
    download_file(data["videos"][0], video_path)

    image_path = None
    if data["images"]:
        ext = Path(urlparse(data["images"][0]).path).suffix or ".jpg"
        image_path = image_dir / f"{sample_id}_source{ext}"
        download_file(data["images"][0], image_path)

    # 添加记录
    sample = {
        "id": sample_id,
        "category": category,
        "subcategory": subcategory,
        "source_listing_id": listing_id,
        "template_video": {
            "path": f"videos/{category}/{sample_id}.mp4",
            **get_video_info(video_path),
            "source": "etsy",
            "source_url": url,
        },
        "source_product": {
            "image_path": f"images/{category}/{sample_id}_source{ext}" if image_path else None,
            "description": data["title"],
        },
        "target_product": {"image_path": None, "description": ""},
        "prompts": {"source": "", "target": ""},
        "difficulty": "medium",
        "shot_type": "pure_product",
        "added_date": datetime.now().strftime("%Y-%m-%d"),
    }
    metadata["samples"].append(sample)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", help="URL 文件")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # 加载 URL
    if args.urls:
        urls = []
        with open(args.urls) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    urls.append((parts[0], parts[1], parts[2]))
                elif len(parts) == 1 and parts[0].startswith("http"):
                    urls.append(("jewelry", "unknown", parts[0]))
    else:
        urls = DEFAULT_URLS

    if args.limit:
        urls = urls[:args.limit]

    print(f"=== PVTT 数据集构建（持久化会话）===")
    print(f"URL: {len(urls)} 个")
    print(f"浏览器数据: {USER_DATA_DIR}")

    metadata = load_metadata()
    print(f"现有样本: {len(metadata['samples'])}\n")

    success = 0
    with sync_playwright() as p:
        # 使用持久化上下文 - 保存 cookies 和登录状态
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=False,  # 必须显示，用户可能需要过验证
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = context.pages[0] if context.pages else context.new_page()

        for i, (cat, subcat, url) in enumerate(urls):
            print(f"\n[{i+1}/{len(urls)}] {cat}/{subcat}")
            print(f"  {url}")

            try:
                if process_listing(page, url, cat, subcat, metadata):
                    success += 1
                    save_metadata(metadata)
            except Exception as e:
                print(f"  错误: {e}")

            time.sleep(3)  # 请求间隔

        context.close()

    print(f"\n=== 完成 ===")
    print(f"成功: {success}/{len(urls)}")
    print(f"总样本: {len(metadata['samples'])}")


if __name__ == "__main__":
    main()
