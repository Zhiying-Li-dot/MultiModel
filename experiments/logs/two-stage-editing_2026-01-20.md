# 两阶段视频编辑实验

**日期**: 2026-01-20
**主题**: 两阶段编辑（先移除再添加）
**测试用例**: bracelet_to_necklace (832x480, 25帧)

---

## 实验背景

假设：直接替换（手链→项链）可能导致残留或冲突，分两步做可能更干净：
1. 先移除原物体
2. 再添加新物体

---

## Stage 0: 生成空背景首帧

### 0.1 生成 Mask (Grounded-SAM-2)

使用 Grounded-SAM-2 + HuggingFace API 检测并分割手链区域：

```bash
python scripts/generate_mask.py \
    --input $CASE_DIR/source_frame1.png \
    --output $CASE_DIR/bracelet_mask.png \
    --prompt "bracelet."
```

对 mask 进行膨胀处理（30x30 kernel）以覆盖边缘：
```python
kernel = np.ones((30, 30), np.uint8)
dilated = cv2.dilate(mask, kernel, iterations=1)
```

### 0.2 LaMa Inpainting 实验

使用 LaMa (Large Mask Inpainting) 进行纹理填充式物体移除。

**脚本用法**：
```bash
python scripts/lama_inpaint.py \
    --input $CASE_DIR/source_frame1.png \
    --mask $CASE_DIR/bracelet_mask.png \
    --output $CASE_DIR/empty_background_lama.png \
    --dilate 30 \
    --mask-mode convex  # original/bbox/convex
```

测试了三种 mask 形状：

| Mask 类型 | 文件 | 说明 |
|----------|------|------|
| 轮廓 (dilated) | `bracelet_mask_dilated.png` | Grounded-SAM-2 分割结果 + 膨胀 |
| Bounding Box | `bracelet_mask_bbox.png` | 包围所有轮廓点的矩形 (530x333) |
| Convex Hull | `bracelet_mask_convex.png` | 所有轮廓点的凸包 |

**LaMa 结果：**

| Mask 类型 | 输出文件 | 效果 |
|----------|---------|------|
| 轮廓 (dilated) | `empty_background_lama.png` | 中间区域有填充痕迹 |
| Bounding Box | `empty_background_lama_bbox.png` | 区域过大 |
| Convex Hull | `empty_background_lama_convex.png` | 待评估 |

**结论**：LaMa 是纯纹理填充（不需要 prompt），适合物体移除。相比 Flux inpainting（prompt-guided，会生成新内容），LaMa 更适合"移除"场景。

---

## 实验配置

### Stage 1: 移除手链

```bash
CASE_DIR=~/pvtt/data/pvtt-benchmark/cases/bracelet_to_necklace

python scripts/ti2v_rfsolver.py \
    --checkpoint-dir /data/xuhao/Wan2.2/Wan2.2-TI2V-5B \
    --source-video $CASE_DIR/source_video.mp4 \
    --target-frame $CASE_DIR/source_frame1.png \
    --source-prompt "A black leather bracelet and silver chain bracelet on purple silk fabric, elegant jewelry display, soft lighting" \
    --target-prompt "Purple silk fabric, elegant jewelry display, soft lighting" \
    --output ~/pvtt/experiments/results/compositional/two_stage_step1_remove.mp4 \
    --max-frames 25 \
    --width 832 --height 480 \
    --steps 50 --cfg 5.0 --shift 0.5
```

### Stage 2: 添加项链

```bash
python scripts/ti2v_rfsolver.py \
    --checkpoint-dir /data/xuhao/Wan2.2/Wan2.2-TI2V-5B \
    --source-video ~/pvtt/experiments/results/compositional/two_stage_step1_remove.mp4 \
    --target-frame $CASE_DIR/target_frame1.png \
    --source-prompt "Purple silk fabric, elegant jewelry display, soft lighting" \
    --target-prompt "A gold chain necklace with white pearl pendants on purple silk fabric, elegant jewelry display, soft lighting" \
    --output ~/pvtt/experiments/results/compositional/two_stage_step2_add.mp4 \
    --max-frames 25 \
    --width 832 --height 480 \
    --steps 50 --cfg 5.0 --shift 0.5
```

---

## Prompts 汇总

| 阶段 | Source Prompt | Target Prompt |
|------|---------------|---------------|
| Stage 1 | A black leather bracelet and silver chain bracelet on purple silk fabric, elegant jewelry display, soft lighting | Purple silk fabric, elegant jewelry display, soft lighting |
| Stage 2 | Purple silk fabric, elegant jewelry display, soft lighting | A gold chain necklace with white pearl pendants on purple silk fabric, elegant jewelry display, soft lighting |

---

## 实验结果

### Stage 1: 移除手链

| 指标 | 值 |
|------|---|
| Inverted std | |
| 视觉质量 | |

### Stage 2: 添加项链

| 指标 | 值 |
|------|---|
| Inverted std | |
| 视觉质量 | |

---

## 对比：单阶段 vs 两阶段

| 方法 | 结果 |
|------|------|
| 单阶段（手链→项链） | |
| 两阶段（手链→空→项链） | |

---

## 结论

（待填写）

---

## 相关文件

### 脚本
- Mask 生成: `baseline/compositional-flux-ti2v/scripts/generate_mask.py`
- LaMa Inpainting: `baseline/compositional-flux-ti2v/scripts/lama_inpaint.py`

### Stage 0 (LaMa Inpainting)
- 原始图像: `data/pvtt-benchmark/cases/bracelet_to_necklace/source_frame1.png`
- Mask (轮廓): `data/pvtt-benchmark/cases/bracelet_to_necklace/bracelet_mask_dilated.png`
- Mask (bbox): `data/pvtt-benchmark/cases/bracelet_to_necklace/bracelet_mask_bbox.png`
- Mask (凸包): `data/pvtt-benchmark/cases/bracelet_to_necklace/bracelet_mask_convex.png`
- LaMa 输出 (轮廓): `data/pvtt-benchmark/cases/bracelet_to_necklace/empty_background_lama.png`
- LaMa 输出 (bbox): `data/pvtt-benchmark/cases/bracelet_to_necklace/empty_background_lama_bbox.png`
- LaMa 输出 (凸包): `data/pvtt-benchmark/cases/bracelet_to_necklace/empty_background_lama_convex.png`

### Stage 1 & 2 (视频编辑)
- Stage 1 配置: `data/pvtt-benchmark/cases/bracelet_to_necklace/config_stage1_remove.yaml`
- Stage 2 配置: `data/pvtt-benchmark/cases/bracelet_to_necklace/config_stage2_add.yaml`
- Stage 1 输出: `experiments/results/compositional/two_stage_step1_remove.mp4`
- Stage 2 输出: `experiments/results/compositional/two_stage_step2_add.mp4`
