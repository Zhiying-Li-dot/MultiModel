# 实验运行指南

本文档说明如何运行 PVTT 项目中的各个实验脚本。

## 环境准备

### 5090 服务器

所有 GPU 计算任务在 5090 服务器上运行（8x RTX 5090 32GB）。

```bash
# SSH 连接
ssh 5090

# 设置 HuggingFace 镜像（中国网络必需）
export HF_ENDPOINT=https://hf-mirror.com

# 使用 wan conda 环境
conda activate wan
# 或直接使用完整路径
~/.conda/envs/wan/bin/python
```

### 本地环境

图像编辑任务（Flux.2）可在本地运行，需要 FAL API：

```bash
pip install fal-client pillow

# 设置 API Key
export FAL_KEY="your-fal-api-key"
```

---

## 方法一：Flux.2 + TI2V 组合方法（当前最佳）

### 完整 Pipeline

```bash
cd ~/pvtt/baseline/compositional-flux-ti2v
python run_pipeline.py --config config/bracelet_to_necklace.yaml
```

### 分步运行

**Step 1: 提取模板视频首帧**

```bash
python scripts/extract_frame.py \
    --video ../../data/videos/bracelet_shot1.mp4 \
    --output ./inputs/bracelet_frame1.png
```

**Step 2: Flux.2 图像编辑（本地）**

```bash
python scripts/flux_edit.py \
    --template-frame ./inputs/bracelet_frame1.png \
    --product-image ../../data/images/necklace.jpg \
    --source-object "bracelets" \
    --target-object "gold necklace with pearls" \
    --output ./results/target_frame1.png
```

**Step 3: TI2V 视频生成（5090 服务器）**

```bash
ssh 5090
cd ~/pvtt/baseline/compositional-flux-ti2v
~/.conda/envs/wan/bin/python scripts/ti2v_generate.py \
    --first-frame ./results/target_frame1.png \
    --prompt "A gold necklace with pearls on purple silk, elegant jewelry display" \
    --output ./results/target_video.mp4
```

---

## 方法二：FlowAlign / FlowEdit（Wan2.1）

### 运行 FlowAlign（推荐）

```bash
ssh 5090
cd ~/pvtt/baseline/flowedit-wan2.1
~/.conda/envs/wan/bin/python awesome_wan_editing.py \
    --config ./config/pvtt/bracelet_flowalign.yaml
```

### 运行 FlowEdit

```bash
~/.conda/envs/wan/bin/python awesome_wan_editing.py \
    --config ./config/pvtt/bracelet_flowedit.yaml
```

### 配置文件说明

```yaml
# config/pvtt/bracelet_flowalign.yaml
video:
    video_path: ./videos/bracelet_shot1.mp4
    source_prompt: "A woman's hand wearing beaded bracelets..."
    target_prompt: "A woman's hand wearing a gold chain necklace..."
    source_blend: bracelets
    target_blend: necklace

training-free-type:
    flag_flowedit: False      # FlowEdit 2-branch
    flag_flowalign: True      # FlowAlign 3-branch（推荐）

flowalign:
    strength: 0.7             # 编辑强度 (0-1)
    target_guidance_scale: 19.5
    flag_attnmask: True       # MasaCtrl masking
    zeta_scale: 1e-3          # 正则化系数
    save_video: ./results/pvtt/output.mp4
```

### 常用配置

| 配置文件 | 说明 |
|----------|------|
| `bracelet_flowalign.yaml` | FlowAlign + MasaCtrl（效果最好） |
| `bracelet_flowedit.yaml` | 纯 FlowEdit |
| `bracelet_wanalign.yaml` | WANAlign |
| `pillow_flowedit.yaml` | 枕套案例 |

---

## 方法三：TI2V + FlowEdit（Wan2.2）

### 无图像条件（等同于 Wan2.1 FlowEdit）

```bash
ssh 5090
cd ~/pvtt/baseline/flowedit-wan2.2
~/.conda/envs/wan/bin/python flowalign_t2v.py \
    --config ../flowedit-wan2.1/config/pvtt/bracelet_flowedit.yaml
```

### 有图像条件（实验性，后续帧会退化）

```bash
cd ~/pvtt/baseline/compositional-flux-ti2v
~/.conda/envs/wan/bin/python scripts/ti2v_flowedit.py \
    --source-video ../../data/videos/bracelet_shot1.mp4 \
    --source-prompt "A woman's hand wearing beaded bracelets..." \
    --target-prompt "A woman's hand wearing a pearl necklace..." \
    --target-first-frame ./results/target_frame1.png \
    --output ./results/ti2v_flowedit.mp4 \
    --use-image-cond
```

> **注意**：有图像条件模式存在理论问题（Inversion-Free 导致后续帧退化），仅供实验验证。

---

## 工具脚本

### 视频对比

```bash
# 三路对比
python scripts/create_compositional_comparison.py \
    --video1 results/source.mp4 \
    --video2 results/flowalign.mp4 \
    --video3 results/flux_ti2v.mp4 \
    --output results/comparison.mp4
```

### 提取帧

```bash
python scripts/extract_frame.py \
    --video input.mp4 \
    --frame 0 \
    --output frame0.png
```

---

## 常见问题

### 1. CUDA Out of Memory

```bash
# 使用更小的分辨率
--size 480x272

# 或减少帧数
--num-frames 25
```

### 2. HuggingFace 下载失败

```bash
# 确保设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 模型路径

5090 服务器上的模型位置：
- Wan2.1: `~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B`
- Wan2.2: `/data/xuhao/Wan2.2/`

---

## 实验结果目录

```
experiments/results/
├── flowedit-wan2.1/        # Wan2.1 FlowEdit/FlowAlign 结果
├── flowedit-wan2.2/        # Wan2.2 结果
└── compositional/          # Flux.2 + TI2V 结果
    ├── target_frame1.png   # Flux.2 生成的首帧
    ├── target_video.mp4    # TI2V 生成的视频
    └── comparison.mp4      # 对比视频
```
