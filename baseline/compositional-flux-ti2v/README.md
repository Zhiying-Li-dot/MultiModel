# Compositional Flux + TI2V Baseline

组合式方法：Flux.2 Dev Edit (图像编辑) + Wan2.1 TI2V (视频生成)

## 方法概述

```
模板视频第一帧 ──┐
                ├──▶ Flux.2 Dev Edit ──▶ 目标第一帧 ──▶ Wan2.1 TI2V ──▶ 目标视频
目标产品图片 ───┘
```

## 环境配置

```bash
# 安装依赖
pip install fal-client pillow

# 设置 API Key
export FAL_KEY="your-fal-api-key"
```

## 使用方法

### Stage 1: 图像编辑

```bash
python scripts/flux_edit.py \
    --template-frame ./inputs/bracelet_frame1.png \
    --product-image ./inputs/necklace.jpg \
    --source-object "bracelets" \
    --target-object "gold necklace with pearls" \
    --output ./results/target_frame1.png
```

### Stage 2: 视频生成

```bash
# 在 5090 服务器运行
python scripts/ti2v_generate.py \
    --first-frame ./results/target_frame1.png \
    --prompt "A gold necklace with pearls on purple silk..." \
    --output ./results/target_video.mp4
```

### 完整 Pipeline

```bash
python run_pipeline.py --config config/bracelet_to_necklace.yaml
```

## 目录结构

```
compositional-flux-ti2v/
├── README.md
├── run_pipeline.py          # 完整 pipeline
├── scripts/
│   ├── flux_edit.py         # Stage 1: Flux.2 图像编辑
│   ├── ti2v_generate.py     # Stage 2: TI2V 视频生成
│   └── extract_frame.py     # 提取视频第一帧
├── config/
│   └── bracelet_to_necklace.yaml
└── results/
```

## 配置文件格式

```yaml
# config/bracelet_to_necklace.yaml
template:
  video_path: ../../data/videos/bracelet_shot1.mp4
  source_object: "bracelets"

product:
  image_path: ../../data/images/JEWE005_source.jpg
  target_object: "gold chain necklace with white pearl pendants"

output:
  first_frame: ./results/bracelet_to_necklace_frame1.png
  video: ./results/bracelet_to_necklace.mp4

flux_edit:
  guidance_scale: 3.5
  num_inference_steps: 28

ti2v:
  num_frames: 49
  guidance_scale: 5.0
```
