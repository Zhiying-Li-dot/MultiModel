# PVTT Benchmark Dataset Statistics

**生成时间:** 2026-01-07
**数据集版本:** 2.0
**总样本数:** 53

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 总样本数 | 53 |
| 视频总大小 | 87 MB |
| 图片总大小 | 29 MB |
| 数据集总大小 | 117 MB |
| 平均视频大小 | 1.64 MB |
| 平均图片大小 | 0.55 MB |
| 收集天数 | 2 天 (2026-01-06 至 2026-01-07) |

## 🗂️ 品类分布

按数据集标准品类统计：

| 品类 | 样本数 | 占比 | ID 范围 |
|------|--------|------|---------|
| Toys & Games | 15 | 28.3% | TOYS001-015 |
| Jewelry | 8 | 15.1% | JEWE001-006, JEW001-002 |
| Home & Living | 8 | 15.1% | HOME001-008 |
| Clothing | 8 | 15.1% | CLOT001-008 |
| Electronics & Accessories | 3 | 5.7% | ELEC001-003 |
| Shoes | 3 | 5.7% | SHOE001-003 |
| Art & Collectibles | 3 | 5.7% | ART001-003 |
| Craft Supplies & Tools | 2 | 3.8% | CRAF001-002 |
| Pet Supplies | 1 | 1.9% | PET001 |
| Bags & Purses | 1 | 1.9% | ACCE001 |
| Books, Movies & Music | 1 | 1.9% | MEDI001 |

## 🏷️ Etsy 原始分类分布

按 Etsy 一级分类统计：

| Etsy 分类 | 样本数 | 占比 |
|-----------|--------|------|
| Toys & Games | 15 | 28.3% |
| Clothing | 8 | 15.1% |
| Home & Living | 7 | 13.2% |
| Jewelry | 6 | 11.3% |
| Electronics & Accessories | 3 | 5.7% |
| Shoes | 3 | 5.7% |
| Art & Collectibles | 3 | 5.7% |
| Craft Supplies & Tools | 2 | 3.8% |
| Pet Supplies | 1 | 1.9% |
| Bags & Purses | 1 | 1.9% |
| Books, Movies & Music | 1 | 1.9% |

## 📅 收集时间线

| 日期 | 新增样本数 | 累计样本数 |
|------|-----------|-----------|
| 2026-01-06 | 3 | 3 |
| 2026-01-07 | 50 | 53 |

## 🔍 数据来源

| 来源 | 样本数 | 占比 |
|------|--------|------|
| Etsy | 53 | 100% |

## 📁 文件组织

```
data/pvtt-benchmark/
├── videos/              # 87 MB
│   ├── toys/           # 15 个视频
│   ├── jewelry/        # 8 个视频
│   ├── home/           # 8 个视频
│   ├── clothing/       # 8 个视频
│   ├── electronics/    # 3 个视频
│   ├── shoes/          # 3 个视频
│   ├── art/            # 3 个视频
│   ├── craft/          # 2 个视频
│   ├── pet/            # 1 个视频
│   ├── accessories/    # 1 个视频
│   └── media/          # 1 个视频
├── images/              # 29 MB
│   └── (同上品类结构)
└── annotations/
    ├── metadata.json    # 主元数据文件
    └── statistics.json  # 统计数据（机器可读）
```

## 🛠️ 收集方法

- **工具:** Chrome 浏览器扩展程序（PVTT Data Collector）
- **自动化程度:** 半自动（人工浏览 + 一键提取）
- **质量控制:** 100% 人工审核
- **数据完整性:** 100%（所有样本包含视频、图片、元数据）

## 📝 元数据字段

每个样本包含以下信息：

- `id`: 数据集内唯一标识符
- `category`: 数据集标准品类
- `listing_id`: Etsy 商品 ID
- `product_handle`: Etsy 商品 URL 标识
- `etsy_taxonomy`: Etsy 完整分类路径（3-5级）
- `template_video`: 视频文件信息（路径、来源、URL）
- `source_product`: 源商品图片和描述
- `added_date`: 添加到数据集的日期
- `download_date`: 从 Etsy 下载的日期
- `notes`: 备注信息

## 🎯 下一步计划

### 数据收集目标

- **当前进度:** 53 / 100+ 样本（53%）
- **优先扩充品类:**
  - Pet Supplies (1 → 5+)
  - Books, Movies & Music (1 → 5+)
  - Bags & Purses (1 → 5+)
  - Craft Supplies (2 → 5+)

### 质量提升

- 平衡各品类样本数量
- 确保多样化的产品类型
- 验证视频质量和可用性

### 实验验证

- 使用现有 53 个样本运行 FlowAlign baseline
- 评估不同品类的迁移效果
- 识别困难样本和改进方向
