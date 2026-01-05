# 实验记录

## 实验列表

| 日期 | 实验 | 方法 | 结果 |
|------|------|------|------|
| 2026-01-05 | [FlowAlign with Wan2.2 TI2V-5B](#flowalign-wan22-2026-01-05) | Wan2.2 TI2V-5B | ✅ T2V模式成功 |
| 2026-01-04 | [Baseline: 手链→项链](#baseline-2026-01-04) | WANAlign2.1 | ⚠️ 部分成功 |

---

## FlowAlign with Wan2.2 TI2V-5B (2026-01-05)

### 实验目标

将FlowAlign从Wan2.1迁移到Wan2.2 TI2V-5B，验证算法兼容性，并尝试添加图像条件支持。

### 实验过程

#### 1. 尝试Wan2.1 I2V-14B（失败）

- **问题**：架构不兼容，I2V模型使用CLIP image encoder，与T2V的text-only架构不同
- **结论**：无法直接将FlowAlign应用于I2V模型

#### 2. 尝试Wan2.2 TI2V-5B + diffusers（失败）

- **问题**：OOM错误，diffusers的CPU offload机制与手动VAE操作冲突
- **结论**：需要使用官方Wan2.2代码而非diffusers wrapper

#### 3. 使用官方Wan2.2代码实现FlowAlign（成功）

创建 `wan2.2-official/flowalign_t2v.py`，关键发现：

| 对比项 | flowedit-wan (Wan2.1) | 我们的实现 (Wan2.2) |
|--------|----------------------|---------------------|
| 模型 | T2V-14B (z_dim=16) | TI2V-5B (z_dim=48) |
| shift | 3.0 | 3.0 (需要与diffusers一致) |
| 噪声策略 | 每步新噪声 | 每步新噪声 (固定噪声效果差) |

### 关键发现

1. **shift参数很重要**
   - Wan2.2官方默认shift=5.0，但FlowAlign需要shift=3.0（与diffusers一致）
   - shift=5.0时编辑效果差，shift=3.0显著改善

2. **每步新噪声更好**
   - 固定噪声：背景保持差
   - 每步新噪声：随机性提供正则化效果，背景保持更好

3. **TI2V的expand_timesteps模式**
   - TI2V使用per-frame timestep：第一帧t=0（固定），其他帧t=t
   - 这可能与FlowAlign的假设冲突，T2V模式更稳定

4. **velocity差异太小的问题**
   - 当source/target prompt语义接近时，velocity差异很小
   - 需要足够的guidance_scale (19.5) 来放大差异

### 最终参数配置

```python
# flowalign_t2v.py 默认参数
shift = 3.0              # 与diffusers一致
strength = 0.7           # 编辑强度
guidance_scale = 19.5    # CFG scale
zeta_scale = 0.001       # 结构保持
steps = 50               # 采样步数
```

### 运行方法

```bash
cd experiments/baseline/wan2.2-official

python flowalign_t2v.py \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --video input.mp4 \
  --source_prompt "description of source" \
  --target_prompt "description of target" \
  --output output.mp4
```

### 输出文件

- `wan2.2-official/results/flowalign_t2v_shift3.mp4` - 最佳结果（shift=3.0）
- `wan2.2-official/results/ti2v_necklace_normal.mp4` - 正常TI2V生成（验证模型正常）

### 待解决问题

1. **TI2V图像条件**：expand_timesteps模式下如何正确注入target image condition
2. **编辑强度**：prompt语义接近时编辑效果有限
3. **模型差异**：TI2V-5B对prompt敏感度可能不如纯T2V模型

---

## Baseline (2026-01-04)

### 实验配置

| 项目 | 值 |
|------|-----|
| 输入视频 | `baseline/bracelet_shot1_480p.mp4` (832×480, 49帧, 3s) |
| Source Prompt | Two personalized couple bracelets, one silver and one black, placed on a purple silk fabric with decorative stones. |
| Target Prompt | A gold charm necklace with colorful gemstone pendants placed on a purple silk fabric with decorative stones. |
| 方法 | WANAlign2.1 (FlowAlign + WAN2.1-1.3B) |
| 参数 | strength=0.7, target_guidance_scale=19.5, flag_attnmask=True, zeta_scale=1e-3 |
| GPU | RTX 5090 32GB |

### 评估结果

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 产品替换 | ✅ 成功 | 手链被替换为项链 |
| 背景保持 | ✅ 成功 | 紫色丝绸背景保持完整 |
| 时序一致性 | ⚠️ 有轻微问题 | 产品有轻微闪烁和抖动 |
| 产品细节 | ⚠️ 清晰度不佳 | 项链细节模糊，宝石不够清晰 |

### 结论

Baseline 验证了预期：
1. ✅ Training-Free 方法可以完成基本的产品替换
2. ✅ 背景保持能力较好
3. ⚠️ **时序一致性**是核心问题 → PVTT 需要解决
4. ⚠️ **产品细节保真度**不足 → PVTT 需要解决

### 输出文件

- 输入：`baseline/bracelet_shot1_480p.mp4`
- 输出：`baseline/flowalign_bracelet_to_necklace.mp4`

---

## 环境配置

### 5090 机器

```bash
# 使用已有的 wan conda 环境
~/.conda/envs/wan/bin/python

# 或安装依赖
pip install torch torchvision torchaudio
pip install omegaconf imageio imageio-ffmpeg matplotlib ftfy
pip install transformers==4.51.3

# 安装 custom diffusers
cd experiments/baseline/flowedit-wan/diffusers
pip install -e .
```

### 运行实验

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd experiments/baseline/flowedit-wan
python awesome_wan_editing.py --config=./config/pvtt/bracelet_to_necklace.yaml
```
