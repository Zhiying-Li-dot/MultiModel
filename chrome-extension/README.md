# PVTT Data Collector - Chrome Extension

一键从 Etsy 提取产品视频和图片，构建 PVTT 数据集。

## 功能

- ✅ 自动提取 Etsy 商品页的视频和所有图片
- ✅ 自动识别 Etsy 分类路径（Jewelry > Earrings > Dangle & Drop Earrings）
- ✅ 一键下载到本地，按日期和商品组织
- ✅ 为每个商品生成独立的 metadata.json
- ✅ 在真实浏览器中运行，不会被反爬检测
- ✅ 支持重复下载覆盖（方便更新数据）

## 安装

### 1. 创建图标（临时方案）

在安装前需要创建 PNG 图标：

```bash
cd /Users/verypro/research/pvtt/chrome-extension

# 创建简单的蓝色方块作为临时图标
python3 << 'EOF'
from PIL import Image, ImageDraw

for size in [16, 48, 128]:
    img = Image.new('RGB', (size, size), '#0066cc')
    draw = ImageDraw.Draw(img)
    # 添加白色边框
    draw.rectangle([2, 2, size-3, size-3], outline='white', width=2)
    img.save(f'icon{size}.png')
    print(f'Created icon{size}.png')
EOF
```

如果没有 PIL，安装：`pip install Pillow`

### 2. 加载到 Chrome

1. 打开 Chrome，访问 `chrome://extensions/`
2. 开启右上角的"开发者模式"
3. 点击"加载已解压的扩展程序"
4. 选择 `/Users/verypro/research/pvtt/chrome-extension` 目录
5. 插件加载完成！

## 使用方法

### 快速开始

1. **打开 Etsy 商品页**（必须包含视频）

   示例：https://www.etsy.com/listing/4412914486/dainty-birthstone-earrings-diamond

2. **点击浏览器工具栏的 🎬 图标**

3. **查看提取的信息**

   插件会自动显示：
   - Listing ID: `4412914486`
   - Title: `Dainty Birthstone Earrings...`
   - Category Path: `Jewelry > Earrings > Dangle & Drop Earrings`
   - Videos (1): 视频URL
   - Images (8): 所有商品图片URL

4. **点击 "📥 Download & Add to Dataset"**

5. **文件自动下载到本地**

   ```
   ~/Downloads/PVTT/2026-01-06/4412914486_dainty-birthstone-earrings-diamond/
   ├── video.mp4          # 商品视频
   ├── image_1.jpg        # 第1张图片
   ├── image_2.jpg        # 第2张图片
   ├── ...                # 更多图片
   └── metadata.json      # 商品元数据
   ```

### 批量收集

重复以上步骤，每个商品会保存在独立的文件夹中：

```
~/Downloads/PVTT/
├── 2026-01-06/
│   ├── 4412914486_dainty-birthstone-earrings-diamond/
│   ├── 4419856766_double-chain-gold-anklet/
│   └── 1289719578_personalized-name-necklace/
└── 2026-01-07/
    └── 4387650123_minimalist-pearl-bracelet/
```

每个文件夹包含该商品的所有媒体文件和元数据，完全独立。

## Metadata 格式

每个商品的 `metadata.json` 包含：

```json
{
  "listing_id": "4412914486",
  "product_handle": "dainty-birthstone-earrings-diamond",
  "url": "https://www.etsy.com/listing/4412914486/dainty-birthstone-earrings-diamond",
  "title": "Dainty Birthstone Earrings, Diamond Dangle Hoop Earring...",
  "etsy_taxonomy": ["Jewelry", "Earrings", "Dangle & Drop Earrings"],
  "download_date": "2026-01-06T14:28:45.200Z",
  "video": {
    "filename": "video.mp4",
    "source_url": "https://v.etsystatic.com/video/upload/..."
  },
  "images": [
    {
      "filename": "image_1.jpg",
      "source_url": "https://i.etsystatic.com/..."
    },
    ...
  ]
}
```

所有信息忠实记录 Etsy 原始数据，无任何修改。

## 文件结构

```
chrome-extension/
├── manifest.json       # 扩展配置
├── content.js          # 页面数据提取
├── popup.html          # 弹窗界面
├── popup.js            # 弹窗逻辑
├── background.js       # 后台下载处理
├── icon16.png          # 16x16 图标
├── icon48.png          # 48x48 图标
├── icon128.png         # 128x128 图标
└── README.md           # 本文件
```

## 故障排除

### 插件问题

**Q: 点击插件没反应？**
- 刷新 Etsy 商品页后再试
- 检查浏览器控制台（F12）查看错误

**Q: 提示 "Not a listing page"？**
- 确保 URL 格式为：`https://www.etsy.com/listing/数字/商品名称`
- 不支持分类页、搜索页、首页

**Q: 提示 "No videos found"？**
- 该商品确实没有视频
- 换一个有视频的商品（商品页面能看到视频播放器）

**Q: 提示 "Missing product handle"？**
- URL 格式不完整，缺少商品名称部分
- 刷新页面后重试

### 下载问题

**Q: 下载失败或文件不完整？**
- 检查 Chrome 下载设置（chrome://settings/downloads）
- 确保没有禁用自动下载
- 检查磁盘空间是否充足

**Q: 重复下载会创建 (1), (2) 副本？**
- 刷新扩展：访问 `chrome://extensions/`，点击扩展的刷新按钮
- `conflictAction: 'overwrite'` 应该会覆盖已存在的文件

**Q: 只下载了部分图片？**
- 正常，插件只提取商品轮播区的图片（排除推荐商品的图片）
- 如果商品真的有很多图片但只下载了几张，可能是 Etsy 改了页面结构

### 数据问题

**Q: Category Path 为空？**
- Etsy 页面可能没有面包屑导航
- 刷新页面后重试
- 某些特殊商品可能缺少分类信息

**Q: 图片是 8 张，但 metadata 说有更多？**
- 检查 `etsy_taxonomy` 字段确认是正确的商品
- 可能是同一张图片的不同尺寸被误识别

## 技术细节

### 数据提取方式

- **视频**：从 HTML 中匹配 `"contentURL"` 字段提取
- **图片**：从 `data-src-zoom-image` 属性提取（商品轮播区专用）
- **分类**：从 `nav[aria-label="breadcrumb"]` 面包屑导航提取
- **商品信息**：从 URL、标题等元素提取

### 文件命名

- **视频**：`video.mp4`（固定名称）
- **图片**：`image_1.jpg`, `image_2.jpg`, ... （按顺序编号）
- **扩展名检测**：支持 `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`

### 路径组织

- **日期**：`YYYY-MM-DD` 格式（下载当天的日期）
- **商品文件夹**：`{listing_id}_{product_handle}`
- **文件系统安全**：自动清理 `/\?%*:|"<>` 等非法字符

### 错误处理

- 所有必需字段都有验证（listing_id, product_handle, videos, breadcrumbs, title）
- Chrome API 错误会显示在插件界面
- 下载失败会在后台日志中记录（F12 > Service Worker > Console）

## 下一步

1. **收集 10-20 个首饰类样本**
2. **扩展到其他品类**（家居、美妆、服饰）
3. **目标：100+ 样本**用于 PVTT Benchmark
4. **开始运行 FlowEdit baseline 实验**
