# PVTT 数据处理脚本

## Etsy 数据抓取

### 方法 1: Playwright 自动抓取（推荐）

```bash
# 安装依赖
pip install playwright requests
playwright install chromium

# 抓取单个商品
python scrape_etsy.py https://www.etsy.com/listing/489997879/family-charm-necklace

# 批量抓取
python scrape_etsy.py urls.txt --output ./downloads/

# 调试模式（显示浏览器）
python scrape_etsy.py URL --visible
```

### 方法 2: 浏览器书签 + 手动下载

当 Playwright 被 Cloudflare 拦截时使用。

**步骤 1**: 创建书签
1. 在浏览器中新建书签
2. 将以下代码复制到书签的 URL 字段：

```javascript
javascript:(function(){var imgs=[],vids=[];document.querySelectorAll('img[src*="etsystatic"]').forEach(i=>{var s=i.src.replace(/il_\d+x\d+/,'il_1588xN').replace(/il_\d+xN/,'il_1588xN');if(s.includes('il_')&&!imgs.includes(s))imgs.push(s)});document.querySelectorAll('video source, video[src]').forEach(v=>{var s=v.src||v.getAttribute('src');if(s&&s.includes('etsystatic')&&!vids.includes(s))vids.push(s)});var r='IMAGES:\n'+imgs.join('\n')+'\n\nVIDEOS:\n'+vids.join('\n');console.log(r);prompt('复制以下URL (Cmd+A, Cmd+C):',r);})();
```

**步骤 2**: 提取 URL
1. 打开 Etsy 商品页面
2. 点击书签，URL 会复制到剪贴板

**步骤 3**: 下载文件
```bash
# 从剪贴板下载 (macOS)
pbpaste | python download_etsy_urls.py - --output ./downloads/

# 从文件下载
python download_etsy_urls.py urls.json --output ./downloads/
```

### 方法 3: 手动保存 HTML

```bash
# 1. 在浏览器中打开商品页
# 2. Cmd+S 保存为 .html 文件
# 3. 用脚本解析

python parse_etsy_html.py saved_page.html --download --output ./downloads/
```

---

## 视频处理

### 提取视频元数据

```bash
# 单个文件
python extract_video_metadata.py video.mp4

# 整个目录
python extract_video_metadata.py videos/ --output metadata.json
```

### 镜头检测

```bash
# 安装依赖
pip install scenedetect[opencv]

# 检测镜头
python detect_shots.py video.mp4 --output shots.json

# 调整灵敏度（越低越敏感）
python detect_shots.py video.mp4 --threshold 20.0
```

---

## 数据集管理

### 添加样本到数据集

```bash
# 交互模式
python add_sample.py --interactive

# 命令行模式
python add_sample.py \
    --video video.mp4 \
    --source-image source.jpg \
    --target-image target.jpg \
    --category jewelry \
    --subcategory bracelet \
    --source-prompt "Two bracelets on silk..." \
    --target-prompt "A gold necklace on silk..."
```

---

## 依赖安装

```bash
pip install requests playwright scenedetect[opencv]
playwright install chromium
```
