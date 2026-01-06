#!/usr/bin/env python3
"""
PVTT 数据集一键构建脚本

自动从 Etsy 抓取商品视频和图片，构建 PVTT Benchmark 数据集。

用法:
    # 使用默认 URL 列表
    python build_dataset.py

    # 指定 URL 文件
    python build_dataset.py --urls jewelry_urls.txt

    # 只抓取前 N 个
    python build_dataset.py --limit 5

    # 显示浏览器（调试）
    python build_dataset.py --visible
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

# 品类前缀
CATEGORY_PREFIX = {
    "jewelry": "JEW",
    "home": "HOME",
    "beauty": "BEA",
    "fashion": "FASH",
    "electronics": "ELEC",
}

# 默认 Etsy URL 列表（首饰类）
DEFAULT_URLS = [
    # 项链
    ("jewelry", "necklace", "https://www.etsy.com/listing/489997879/family-charm-necklace-custom-birthstone"),
    ("jewelry", "necklace", "https://www.etsy.com/listing/1128997665/custom-birth-flower-necklace-flower"),
    ("jewelry", "necklace", "https://www.etsy.com/listing/1450227638/personalized-name-necklace-custom-name"),
    # 手链
    ("jewelry", "bracelet", "https://www.etsy.com/listing/615384498/personalized-couple-bracelets-set-two"),
    ("jewelry", "bracelet", "https://www.etsy.com/listing/1074193270/custom-photo-projection-bracelet"),
    # 戒指
    ("jewelry", "ring", "https://www.etsy.com/listing/1233697695/custom-birth-flower-ring-personalized"),
    ("jewelry", "ring", "https://www.etsy.com/listing/1055841862/custom-name-ring-personalized-name-ring"),
    # 耳环
    ("jewelry", "earring", "https://www.etsy.com/listing/1140520285/custom-birth-flower-earrings"),
    # 家居类
    ("home", "pillow", "https://www.etsy.com/listing/1138633008/winter-ski-gear-pillow-cover-12x20-inch"),
    ("home", "decor", "https://www.etsy.com/listing/1567890123/custom-family-photo-frame"),
]


def load_metadata() -> dict:
    """加载现有 metadata"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": "1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "description": "PVTT Benchmark Dataset",
        "samples": []
    }


def save_metadata(metadata: dict):
    """保存 metadata"""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_next_id(metadata: dict, category: str) -> str:
    """获取下一个可用 ID"""
    prefix = CATEGORY_PREFIX.get(category, "UNK")
    existing = [s["id"] for s in metadata["samples"] if s["id"].startswith(prefix)]

    if not existing:
        return f"{prefix}001"

    max_num = max(int(id_[len(prefix):]) for id_ in existing)
    return f"{prefix}{max_num + 1:03d}"


def extract_listing_id(url: str) -> str:
    """从 URL 提取 listing ID"""
    match = re.search(r'/listing/(\d+)', url)
    return match.group(1) if match else "unknown"


def scrape_etsy_page(page, url: str) -> dict:
    """抓取单个 Etsy 页面"""
    listing_id = extract_listing_id(url)

    print(f"  加载页面...")
    try:
        page.goto(url, timeout=60000)
        # 等待页面实际加载（检测商品标题元素）
        page.wait_for_selector("h1", timeout=30000)
        print(f"  页面加载成功")
    except PlaywrightTimeout:
        print(f"  页面加载超时")
        # 可能需要处理 Cloudflare challenge
        time.sleep(10)
    except Exception as e:
        print(f"  页面加载错误: {e}")
        time.sleep(5)

    # 等待页面稳定并模拟用户行为
    time.sleep(2)
    try:
        page.mouse.move(500, 400)
        time.sleep(1)
        page.evaluate('window.scrollTo(0, 300)')
        time.sleep(2)
    except:
        pass

    # 获取标题
    title = ""
    try:
        title_el = page.query_selector("h1")
        if title_el:
            title = title_el.inner_text().strip()
    except:
        pass

    # 提取媒体 URL
    images = set()
    videos = set()

    try:
        content = page.content()
        print(f"  页面大小: {len(content)} bytes")

        # 从 JSON-LD 提取 contentURL（最可靠的方法）
        content_urls = re.findall(r'"contentURL"\s*:\s*"([^"]+)"', content)
        print(f"  找到 {len(content_urls)} 个 contentURL")

        for url_escaped in content_urls:
            url_clean = url_escaped.replace('\\/', '/')
            if 'v.etsystatic.com' in url_clean:
                # 视频 URL，添加 .mp4 后缀
                if not url_clean.endswith('.mp4'):
                    url_clean += '.mp4'
                videos.add(url_clean)
                print(f"  发现视频: {url_clean[:60]}...")
            elif 'i.etsystatic.com' in url_clean and 'il_' in url_clean:
                images.add(url_clean)

        # 备用：从页面内容提取高清图片
        img_patterns = [
            r'https://i\.etsystatic\.com/\d+/r/il/[a-f0-9]+/\d+/il_fullxfull\.[^"\'>\s\\]+\.(?:jpg|png|webp)',
            r'https://i\.etsystatic\.com/\d+/r/il/[a-f0-9]+/\d+/il_1588xN\.[^"\'>\s\\]+\.(?:jpg|png|webp)',
        ]
        for pattern in img_patterns:
            for match in re.findall(pattern, content):
                images.add(match)

        # 备用：视频 URL（带 .mp4）
        video_patterns = [
            r'https://v\.etsystatic\.com/video/upload/[^"\'>\s\\]+\.mp4',
        ]
        for pattern in video_patterns:
            for match in re.findall(pattern, content):
                videos.add(match)
    except Exception as e:
        print(f"  提取媒体时出错: {e}")

    # 过滤
    images = [img for img in images if "icon" not in img.lower() and "avatar" not in img.lower()]

    return {
        "listing_id": listing_id,
        "title": title,
        "url": url,
        "images": sorted(list(images)),
        "videos": sorted(list(videos)),
    }


def download_file(url: str, output_path: Path, desc: str = "") -> bool:
    """下载文件"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://www.etsy.com/",
        }
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"    ✓ {desc} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"    ✗ {desc}: {e}")
        return False


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
    except:
        return {}


def process_listing(page, url: str, category: str, subcategory: str, metadata: dict) -> bool:
    """处理单个商品"""
    listing_id = extract_listing_id(url)

    # 检查是否已存在
    existing_ids = [s.get("source_listing_id") for s in metadata["samples"]]
    if listing_id in existing_ids:
        print(f"  跳过（已存在）: {listing_id}")
        return False

    # 抓取页面
    data = scrape_etsy_page(page, url)

    if not data["videos"]:
        print(f"  跳过（无视频）: {data['title'][:50]}")
        return False

    # 生成样本 ID
    sample_id = get_next_id(metadata, category)
    print(f"  样本 ID: {sample_id}")
    print(f"  标题: {data['title'][:60]}")
    print(f"  图片: {len(data['images'])} 张, 视频: {len(data['videos'])} 个")

    # 创建目录
    video_dir = DATASET_ROOT / "videos" / category
    image_dir = DATASET_ROOT / "images" / category
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 下载视频（取第一个）
    video_url = data["videos"][0]
    video_path = video_dir / f"{sample_id}.mp4"
    print(f"  下载视频...")
    if not download_file(video_url, video_path, video_path.name):
        return False

    # 下载主图（取第一张）
    if data["images"]:
        img_url = data["images"][0]
        ext = Path(urlparse(img_url).path).suffix or ".jpg"
        image_path = image_dir / f"{sample_id}_source{ext}"
        print(f"  下载图片...")
        download_file(img_url, image_path, image_path.name)
    else:
        image_path = None

    # 获取视频信息
    video_info = get_video_info(video_path)

    # 构建样本条目
    sample = {
        "id": sample_id,
        "category": category,
        "subcategory": subcategory,
        "source_listing_id": listing_id,
        "template_video": {
            "path": f"videos/{category}/{sample_id}.mp4",
            **video_info,
            "source": "etsy",
            "source_url": url,
        },
        "source_product": {
            "image_path": f"images/{category}/{sample_id}_source{ext}" if image_path else None,
            "description": data["title"],
        },
        "target_product": {
            "image_path": None,  # 需要后续手动添加
            "description": "",
        },
        "prompts": {
            "source": "",  # 需要后续手动添加
            "target": "",
        },
        "difficulty": "medium",
        "shot_type": "pure_product",  # 默认，需要后续确认
        "notes": f"Auto-collected from Etsy on {datetime.now().strftime('%Y-%m-%d')}",
        "added_date": datetime.now().strftime("%Y-%m-%d"),
    }

    metadata["samples"].append(sample)
    return True


def main():
    parser = argparse.ArgumentParser(description="PVTT 数据集一键构建")
    parser.add_argument("--urls", help="URL 文件（每行: category subcategory url）")
    parser.add_argument("--limit", type=int, default=0, help="限制抓取数量")
    parser.add_argument("--visible", action="store_true", help="显示浏览器窗口")
    parser.add_argument("--category", default="jewelry", help="默认品类")
    args = parser.parse_args()

    # 加载 URL 列表
    if args.urls:
        urls = []
        with open(args.urls, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    urls.append((parts[0], parts[1], parts[2]))
                elif len(parts) == 1 and parts[0].startswith("http"):
                    urls.append((args.category, "unknown", parts[0]))
    else:
        urls = DEFAULT_URLS

    if args.limit > 0:
        urls = urls[:args.limit]

    print(f"=== PVTT 数据集构建 ===")
    print(f"URL 数量: {len(urls)}")
    print(f"输出目录: {DATASET_ROOT}")
    print()

    # 加载现有 metadata
    metadata = load_metadata()
    print(f"现有样本: {len(metadata['samples'])} 个\n")

    # 启动浏览器
    success_count = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.visible,
            args=['--disable-blink-features=AutomationControlled'],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = context.new_page()
        # 移除 webdriver 标志
        page.add_init_script('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')

        for i, (category, subcategory, url) in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] {category}/{subcategory}")
            print(f"  URL: {url}")

            try:
                if process_listing(page, url, category, subcategory, metadata):
                    success_count += 1
                    # 每成功一个就保存
                    save_metadata(metadata)
            except Exception as e:
                print(f"  错误: {e}")

            print()
            time.sleep(2)  # 避免请求过快

        browser.close()

    # 最终保存
    save_metadata(metadata)

    print(f"=== 完成 ===")
    print(f"成功: {success_count}/{len(urls)}")
    print(f"总样本数: {len(metadata['samples'])}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
