# PVTT Data Collector - Chrome Extension

一键从 Etsy 提取产品视频和图片，构建 PVTT 数据集。

## 功能

- ✅ 自动提取 Etsy 商品页的视频和图片 URL
- ✅ 一键下载到本地
- ✅ 自动生成 metadata.json
- ✅ 在真实浏览器中运行，不会被反爬检测
- ✅ 你手动过验证码，插件自动提取数据

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

### 第一次使用

1. 打开任意 Etsy 商品页，例如：
   https://www.etsy.com/listing/4419856766/double-chain-gold-anklet

2. 点击浏览器工具栏的 🎬 图标

3. 插件会自动提取：
   - Listing ID
   - 商品标题
   - 视频 URL（可能有多个）
   - 图片 URL（高清版本）

4. 选择品类（Category）和子品类（Subcategory）

5. 点击 "📥 Download & Add to Dataset"

6. 文件会自动下载到 `~/Downloads/PVTT/` 目录：
   - `PVTT_JEW001.mp4` - 视频
   - `PVTT_JEW001_source.jpg` - 图片
   - `PVTT_metadata.json` - 数据集元数据

### 批量收集

1. 打开第一个商品页，点击插件，下载
2. 打开第二个商品页，点击插件，下载
3. ...重复

metadata.json 会自动累积所有样本。

## 整合到数据集

下载完成后，运行脚本整合到 PVTT 数据集：

```bash
cd /Users/verypro/research/pvtt

# 将下载的文件整合到数据集
python scripts/integrate_extension_data.py ~/Downloads/PVTT/
```

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

**Q: 点击插件没反应？**
A: 刷新 Etsy 商品页后再试

**Q: 提示 "Please open an Etsy listing page"？**
A: 确保当前页面 URL 包含 `/listing/数字`

**Q: 没找到视频？**
A: 不是所有 Etsy 商品都有视频，换一个有视频的商品

**Q: 下载失败？**
A: 检查 Chrome 下载设置，确保没有禁用自动下载

## 技术细节

- **数据提取**：从页面的 JSON-LD 结构化数据中提取，最可靠
- **视频 URL**：自动添加 `.mp4` 后缀
- **图片 URL**：自动转换为 `il_fullxfull` 高清版本
- **ID 生成**：自动递增（JEW001, JEW002...）
- **存储**：使用 Chrome Storage API 保存状态

## 下一步

收集 10-20 个样本后，可以开始运行 FlowEdit baseline 实验！
