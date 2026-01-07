#!/usr/bin/env python3
"""
整合从 Chrome Extension 下载的数据到 PVTT Benchmark

从 ~/Downloads/PVTT/ 复制文件到 data/pvtt-benchmark/
并更新 metadata.json
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DOWNLOAD_DIR = Path.home() / "Downloads" / "PVTT"
DATASET_ROOT = PROJECT_ROOT / "data" / "pvtt-benchmark"
METADATA_FILE = DATASET_ROOT / "annotations" / "metadata.json"

# Etsy一级分类到数据集品类的映射
CATEGORY_MAPPING = {
    "Jewelry": "jewelry",
    "Home & Living": "home",
    "Clothing": "clothing",
    "Toys & Games": "toys",
    "Art & Collectibles": "art",
    "Electronics & Accessories": "electronics",
    "Shoes": "shoes",
    "Bags & Purses": "accessories",
    "Craft Supplies & Tools": "craft",
    "Pet Supplies": "pet",
    "Books, Movies & Music": "media"
}


def load_metadata():
    """加载现有 metadata.json"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {
        "version": "2.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "description": "PVTT Benchmark Dataset - Product Video Template Transfer",
        "samples": []
    }


def get_next_id(metadata, category_prefix):
    """获取下一个可用ID"""
    existing = [s["id"] for s in metadata["samples"] if s["id"].startswith(category_prefix)]
    if not existing:
        return f"{category_prefix}001"

    max_num = max([int(id[len(category_prefix):]) for id in existing])
    return f"{category_prefix}{str(max_num + 1).zfill(3)}"


def integrate_sample(product_folder, metadata):
    """整合单个样本到数据集"""
    # 读取商品 metadata
    product_metadata_file = product_folder / "metadata.json"
    if not product_metadata_file.exists():
        print(f"  跳过 {product_folder.name}: 无 metadata.json")
        return None

    with open(product_metadata_file) as f:
        product_data = json.load(f)

    listing_id = product_data["listing_id"]

    # 检查是否已存在
    if any(s.get("listing_id") == listing_id for s in metadata["samples"]):
        print(f"  跳过 {listing_id}: 已存在")
        return None

    # 映射品类
    etsy_taxonomy = product_data.get("etsy_taxonomy", [])
    if not etsy_taxonomy:
        print(f"  跳过 {listing_id}: 无分类信息")
        return None

    etsy_category = etsy_taxonomy[0]
    category = CATEGORY_MAPPING.get(etsy_category, "other")

    # 生成 ID
    category_prefix = category.upper()[:4]
    sample_id = get_next_id(metadata, category_prefix)

    # 复制视频
    video_src = product_folder / "video.mp4"
    if not video_src.exists():
        print(f"  跳过 {listing_id}: 无视频文件")
        return None

    video_dest_dir = DATASET_ROOT / "videos" / category
    video_dest_dir.mkdir(parents=True, exist_ok=True)
    video_dest = video_dest_dir / f"{sample_id}.mp4"
    shutil.copy2(video_src, video_dest)

    # 复制第一张图片作为source
    image_files = sorted(product_folder.glob("image_*"))
    if image_files:
        image_src = image_files[0]
        image_ext = image_src.suffix
        image_dest_dir = DATASET_ROOT / "images" / category
        image_dest_dir.mkdir(parents=True, exist_ok=True)
        image_dest = image_dest_dir / f"{sample_id}_source{image_ext}"
        shutil.copy2(image_src, image_dest)
        image_path = f"images/{category}/{sample_id}_source{image_ext}"
    else:
        image_path = None

    # 构建数据集条目
    sample = {
        "id": sample_id,
        "category": category,
        "listing_id": listing_id,
        "product_handle": product_data.get("product_handle"),
        "etsy_taxonomy": etsy_taxonomy,
        "template_video": {
            "path": f"videos/{category}/{sample_id}.mp4",
            "source": "etsy",
            "source_url": product_data.get("url")
        },
        "source_product": {
            "image_path": image_path,
            "description": product_data.get("title", "")
        },
        "added_date": datetime.now().strftime("%Y-%m-%d"),
        "download_date": product_data.get("download_date"),
        "notes": "Collected via Chrome Extension"
    }

    print(f"  ✓ {sample_id}: {product_data.get('title', '')[:50]}...")
    return sample


def main():
    print("=== 整合下载数据到 PVTT Benchmark ===\n")

    # 加载现有 metadata
    metadata = load_metadata()
    print(f"当前数据集: {len(metadata['samples'])} 个样本\n")

    # 遍历下载的商品
    added = 0
    skipped = 0

    for date_folder in sorted(DOWNLOAD_DIR.iterdir()):
        if not date_folder.is_dir() or date_folder.name.startswith('.'):
            continue

        print(f"处理 {date_folder.name}:")

        for product_folder in sorted(date_folder.iterdir()):
            if not product_folder.is_dir():
                continue

            sample = integrate_sample(product_folder, metadata)
            if sample:
                metadata["samples"].append(sample)
                added += 1
            else:
                skipped += 1

    # 保存更新后的 metadata
    if added > 0:
        METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n=== 完成 ===")
        print(f"新增: {added} 个样本")
        print(f"跳过: {skipped} 个样本")
        print(f"总计: {len(metadata['samples'])} 个样本")
        print(f"\n数据集位置: {DATASET_ROOT}")
    else:
        print("\n没有新样本需要添加")


if __name__ == "__main__":
    main()
