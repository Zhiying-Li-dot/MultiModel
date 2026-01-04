# 实验记录

## 实验列表

| 日期 | 实验 | 方法 | 结果 |
|------|------|------|------|
| 2026-01-04 | [Baseline: 手链→项链](#baseline-2026-01-04) | WANAlign2.1 | ⚠️ 部分成功 |

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
