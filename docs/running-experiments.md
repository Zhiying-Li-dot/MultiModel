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

---

## 标准化案例数据

实验使用标准化的案例数据，位于 `data/pvtt-benchmark/cases/` 目录：

```
data/pvtt-benchmark/cases/bracelet_to_necklace/
├── config.yaml          # 配置和 prompts
├── source_video.mp4     # 原视频 (832x480, 25帧)
├── source_frame1.png    # 原视频首帧
├── target_frame1.png    # 目标首帧 (Flux.2 生成)
└── product_image.jpg    # 产品参考图（项链）
```

**config.yaml 内容：**

```yaml
prompts:
  source: "A black leather bracelet and silver chain bracelet on purple silk fabric, elegant jewelry display, soft lighting"
  target: "A gold chain necklace with white pearl pendants on purple silk fabric, elegant jewelry display, soft lighting"

flux_edit:
  source_object: "black leather bracelet and silver chain bracelet"
  target_object: "gold chain necklace with pearl pendants"

resolution:
  width: 832
  height: 480
```

---

## 方法一：Flux.2 + TI2V 组合方法（当前最佳）

使用已准备好的标准化案例数据：

```bash
# 案例数据路径
CASE_DIR=~/pvtt/data/pvtt-benchmark/cases/bracelet_to_necklace
```

**Step 1: TI2V 视频生成（5090 服务器）**

```bash
ssh 5090
CASE_DIR=~/pvtt/data/pvtt-benchmark/cases/bracelet_to_necklace
cd ~/pvtt/baseline/compositional-flux-ti2v
~/.conda/envs/wan/bin/python scripts/ti2v_generate.py \
    --first-frame $CASE_DIR/target_frame1.png \
    --prompt "A gold chain necklace with white pearl pendants on purple silk fabric, elegant jewelry display, soft lighting" \
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

## 方法三：RF-Solver + TI2V（实验性）

使用 RF-Solver（二阶 Taylor 展开）进行更精确的 Flow Matching Inversion。

> **重要**：分辨率对齐至关重要！源视频、目标首帧必须使用相同分辨率，否则后续帧会严重退化。
>
> 参考论文：[Taming Rectified Flow for Inversion and Editing](https://arxiv.org/abs/2411.04746) (ICML 2025)

```bash
ssh 5090
CASE_DIR=~/pvtt/data/pvtt-benchmark/cases/bracelet_to_necklace
cd ~/pvtt/baseline/compositional-flux-ti2v
~/.conda/envs/wan/bin/python scripts/ti2v_rfsolver.py \
    --checkpoint-dir /data/xuhao/Wan2.2/Wan2.2-TI2V-5B \
    --source-video $CASE_DIR/source_video.mp4 \
    --source-prompt "A black leather bracelet and silver chain bracelet on purple silk fabric, elegant jewelry display, soft lighting" \
    --target-prompt "A gold chain necklace with white pearl pendants on purple silk fabric, elegant jewelry display, soft lighting" \
    --target-frame $CASE_DIR/target_frame1.png \
    --output ./results/ti2v_rfsolver.mp4 \
    --width 832 --height 480 \
    --steps 50 --cfg 5.0 --shift 5.0
```

技术原理见 [Flow Matching Inversion 方案](design/rf-inversion-ti2v.md)。

### Euler vs RF-Solver 对比

| 方法 | Inverted noise std | 后续帧质量 |
|------|-------------------|-----------|
| Euler (一阶) | 0.77 | 退化严重 |
| RF-Solver (二阶) | 0.82 (更接近1.0) | 明显改善 |

---

## 方法四：TI2V + FlowEdit（实验性）

### 无图像条件（等同于 Wan2.1 FlowEdit）

```bash
ssh 5090
cd ~/pvtt/baseline/flowedit-wan2.2
~/.conda/envs/wan/bin/python flowalign_t2v.py \
    --config ../flowedit-wan2.1/config/pvtt/bracelet_flowedit.yaml
```

### 有图像条件（已知问题：后续帧退化）

```bash
CASE_DIR=~/pvtt/data/pvtt-benchmark/cases/bracelet_to_necklace
cd ~/pvtt/baseline/compositional-flux-ti2v
~/.conda/envs/wan/bin/python scripts/ti2v_flowedit.py \
    --checkpoint-dir /data/xuhao/Wan2.2/Wan2.2-TI2V-5B \
    --source-video $CASE_DIR/source_video.mp4 \
    --source-prompt "A black leather bracelet and silver chain bracelet on purple silk fabric, elegant jewelry display, soft lighting" \
    --target-prompt "A gold chain necklace with white pearl pendants on purple silk fabric, elegant jewelry display, soft lighting" \
    --target-frame $CASE_DIR/target_frame1.png \
    --output ./results/ti2v_flowedit.mp4
```

> **注意**：有图像条件模式存在理论问题（Inversion-Free 导致后续帧退化），建议使用方法三。

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
- Wan2.2 TI2V-5B: `/data/xuhao/Wan2.2/Wan2.2-TI2V-5B/`
- Flux.2 Dev: 自动从 HuggingFace 下载（需设置 HF_ENDPOINT 镜像）

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
