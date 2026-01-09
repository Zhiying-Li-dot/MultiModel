# PVTT 实验记录

本目录统一管理所有 PVTT 项目的实验结果、配置、分析报告等。

## 目录结构

```
experiments/
├── README.md                    # 本文件
├── results/                     # 实验结果（视频、图片等）
│   ├── flowalign-wan2.1/       # FlowAlign (Wan2.1) 基线结果
│   │   ├── test01_flowalign_watch_to_bracelet.mp4
│   │   ├── test02_flowalign_tray_to_flowers.mp4
│   │   └── ...
│   ├── flowedit-wan2.1/        # FlowEdit (Wan2.1) 基线结果
│   └── our-method/             # 我们的方法实验结果
├── configs/                     # 实验配置（自动生成）
│   └── flowalign-wan2.1/
│       ├── test01.yaml
│       └── ...
├── logs/                        # 实验日志
│   └── flowalign-wan2.1/
│       ├── test01.log
│       └── ...
└── analysis/                    # 实验分析、对比报告
    ├── baseline-comparison.md
    └── metrics/
        └── flowalign-results.csv
```

## 组织原则

1. **按方法分类** - 每个方法（baseline或我们的方法）有独立子目录
2. **统一命名** - 实验结果文件名包含方法名和转换任务
3. **版本管理** - 重要结果提交到 git，大文件用 git-lfs
4. **可追溯** - 保留配置文件和日志，方便复现

## 命名规范

### 结果视频
格式：`{test_id}_{method}_{source}_to_{target}.mp4`

示例：
- `test01_flowalign_watch_to_bracelet.mp4`
- `test02_our_method_tray_to_flowers.mp4`

### 实验日志
格式：`{test_id}_{method}_{timestamp}.log`

示例：
- `test01_flowalign_20260107_161800.log`

## 当前实验

### FlowAlign-Wan2.1 基线

| 测试ID | 类别 | 转换 | 状态 | 结果文件 |
|--------|------|------|------|---------|
| test01 | Jewelry | Watch → Bracelet | ✅ | test01_flowalign_watch_to_bracelet.mp4 |
| test02 | Home | Tray → Flowers | ✅ | test02_flowalign_tray_to_flowers.mp4 |
| test03 | Toys | Stacker → Toy | ✅ | test03_flowalign_stacker_to_ridetoy.mp4 |
| test04 | Clothing | Socks → Skirt | ✅ | test04_flowalign_socks_to_skirt.mp4 |

**参数设置：**
- Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Frames: 49 (固定取前49帧)
- Resolution: 480p (832x480 或 480x832)
- Steps: 50
- Strength: 0.7
- Guidance Scale: 19.5
- Attention Masking: Layers 11-17

## 添加新实验

1. 在 `baseline/flowedit-wan2.1/run_experiments.py` 中添加实验配置
2. 运行实验：`python run_experiments.py --experiments new_test`
3. 结果自动保存到 `experiments/results/{method}/`
4. 更新本文档的实验记录表

## Reference Attention (RefDrop) 实验

### 实验背景

**目标**：将 RefDrop (NeurIPS 2024) 方法适配到视频编辑场景，用真实产品图片引导生成。

**PVTT 适配关键改动**：
- RefDrop 原文：reference 是生成的图片（参与 denoising）
- 我们的适配：使用真实产品图片的 clean features (t=0)
- 预提取 K_ref, V_ref，所有 denoising step 复用

**核心公式**：
```
X' = c * Attention(Q, K_ref, V_ref) + (1-c) * Attention(Q, K, V)
```

### 实验配置

- 测试用例：Test02 手环到项链 (bracelet → necklace)
- Source: `bracelet_shot1.mp4` (黑银情侣手环)
- Reference Image: `JEWE005_source.jpg` (金色珍珠吊坠项链)
- 基础方法：FlowAlign / FlowEdit (Wan2.1-T2V-1.3B)

### 第一轮实验：错误 Prompt（colorful gemstone）

使用了与参考图不匹配的 prompt：`A gold charm necklace with colorful gemstone pendants...`

| c 值 | 方法 | 效果 | 结果文件 |
|------|------|------|---------|
| - | FlowAlign Baseline | 彩色宝石项链，正常编辑 | test02_baseline_bracelet_to_necklace.mp4 |
| 0.2 | FlowAlign + RefDrop | 有编辑 + **黑色棋盘伪影** | test02_c0.2_fixed.mp4 |
| 0.5 | FlowAlign + RefDrop | 几乎无编辑（源视频） | test02_c0.5_fixed.mp4 |
| 0.8 | FlowAlign + RefDrop | 完全无编辑（源视频） | test02_fixed_c0.8.mp4 |
| - | FlowEdit Baseline | 深绿色珠串+金链 | test02_flowedit_c0.2.mp4 |
| 0.2 | FlowEdit + RefDrop | 几乎无编辑（源视频） | test02_flowedit_refdrop_c0.2.mp4 |
| 0.1 | FlowEdit + RefDrop | 黑丝带+金链+金心吊坠 | test02_flowedit_refdrop_c0.1.mp4 |
| 0.05 | FlowEdit + RefDrop | 彩色宝石+金链 | test02_flowedit_refdrop_c0.05.mp4 |

**问题发现**：参考图是珍珠项链，但生成结果一颗珍珠都没有！

### 第二轮实验：正确 Prompt（white pearl）

修正 prompt 为：`A gold chain necklace with white pearl drop pendants and red gemstone accents...`

| c 值 | 方法 | 珍珠效果 | 结果文件 |
|------|------|---------|---------|
| - | FlowEdit Baseline | ✅ 有珍珠，混有粉色珠子 | test02_flowedit_pearl_baseline.mp4 |
| **0.05** | FlowEdit + RefDrop | ✅✅ **珍珠最多最纯净** | test02_flowedit_pearl_c0.05.mp4 |
| 0.1 | FlowEdit + RefDrop | ✅ 有珍珠+红宝石 | test02_flowedit_pearl_c0.1.mp4 |
| 0.2 | FlowEdit + RefDrop | ❌ 无珍珠（编辑被阻断） | test02_flowedit_pearl_c0.2.mp4 |

**对比视频**: `test02_4way_comparison.mp4` (Source | Reference | Baseline | RefDrop c=0.05)

### 核心发现

#### 1. Prompt 是关键，必须与参考图匹配
- 错误 prompt（colorful gemstone）+ 珍珠参考图 → 生成彩色宝石，无珍珠
- 正确 prompt（white pearl）+ 珍珠参考图 → 生成珍珠项链
- **结论**：文本 prompt 主导生成内容，参考图提供微调

#### 2. c 值对编辑方法敏感
- **FlowAlign**：c=0.2 产生伪影，c≥0.5 完全阻断编辑
- **FlowEdit**：c≥0.2 阻断编辑，c=0.05-0.1 可用
- **结论**：FlowEdit 对 RefDrop 更敏感，需要更小的 c 值

#### 3. RefDrop 与编辑机制冲突
- FlowAlign/FlowEdit 依赖 self-attention 传递时序/编辑信息
- RefDrop 用静态图像特征替换 self-attention 输出
- 高 c 值破坏编辑信号，导致输出接近源视频

#### 4. RefDrop 提供微调作用
- 基线（无RefDrop）+ 正确prompt：珍珠 + 混合粉色珠子
- c=0.05 + 正确prompt：更纯净的珍珠串，更接近参考图
- **结论**：RefDrop 在正确 prompt 基础上提供轻微的正向引导

### 最佳实践

```yaml
# FlowEdit + RefDrop 推荐配置
flowedit:
    use_reference_attention: True
    ref_c: 0.05  # 不要超过 0.1
    # target_prompt 必须与参考图内容匹配！
```

### 第三轮实验：Noisy RefDrop vs Clean RefDrop

**假设**：Clean RefDrop 在 t=0 提取特征，可能与高噪声 timestep 的视频特征分布不匹配。

**Noisy RefDrop 方法**：
- 每个 denoising step，给参考图 latent 添加与当前 timestep 匹配的噪声
- 从 noisy reference 提取 K_ref, V_ref
- 特征分布与当前视频特征匹配

**实现**：
- 分支：`feature/noisy-refdrop`
- 预计算所有 timestep 的 reference features（存储在 CPU，使用时移到 GPU）
- 通过 hook 追踪当前 timestep

| 方法 | c 值 | 珍珠效果 | 结果文件 |
|------|------|---------|---------|
| Clean RefDrop | 0.05 | ✅ 有珍珠 | test02_flowedit_pearl_c0.05.mp4 |
| Noisy RefDrop | 0.2 | ❌ 无珍珠 | test02_noisy_refdrop_c0.2.mp4 |
| Noisy RefDrop | 0.05 | ✅ 有珍珠 | test02_noisy_refdrop_c0.05.mp4 |

**对比视频**: `test02_clean_vs_noisy_comparison.avi` (Source | Target | Clean | Noisy)

**结论**：
- Noisy RefDrop 效果与 Clean RefDrop 基本相同，没有显著提升
- c=0.05 两种方法都能看到珍珠，c=0.2 都无法保持编辑效果
- **假设不成立**：feature distribution mismatch 不是主要问题
- 问题可能在于 self-attention 注入方式本身与 FlowEdit 机制冲突

### 下一步

- [x] 给 FlowEdit 添加 RefDrop 支持
- [x] 尝试更小的 c 值（0.05, 0.1）
- [x] 尝试 Noisy RefDrop（结果：无显著提升）
- [ ] 在更多样本上验证 c=0.05 的效果
- [ ] 考虑自适应 c 值（不同层/timestep 使用不同 c）
- [ ] 考虑替代方案：注入 cross-attention 而非 self-attention
- [ ] 研究 IP-Adapter / ControlNet 等其他图像条件方法

## 实验分析

### 下一步计划
- [ ] 完成更多样本对的测试
- [x] 对比 FlowAlign vs FlowEdit
- [ ] 计算定量指标（CLIP-score, FVD等）
- [ ] 用户研究评估
- [ ] 实现我们的改进方法
