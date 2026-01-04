# PVTT Baseline Experiments

使用 FlowEdit + WAN2.1 作为 Training-Free baseline，测试现有方法在 PVTT 任务上的效果。

## 环境要求

- GPU: NVIDIA 5090 / A100 (建议 40GB+ 显存)
- Python 3.10
- PyTorch 2.4.0
- CUDA 12.1

## 快速开始

### 1. 在 5090 机器上设置环境

```bash
cd experiments/baseline
chmod +x setup_5090.sh
./setup_5090.sh
```

### 2. 运行实验

```bash
conda activate wanalign
chmod +x run_pvtt_baseline.sh
./run_pvtt_baseline.sh
```

或者手动运行：

```bash
conda activate wanalign
cd flowedit-wan
python awesome_wan_editing.py --config=./config/pvtt/bracelet_to_necklace.yaml
```

## 实验配置

### 测试案例：手链 → 项链

| 项目 | 值 |
|------|-----|
| 输入视频 | `videos/bracelet_shot1.mp4` (6.2s, 镜头1) |
| Source Prompt | Two personalized couple bracelets... |
| Target Prompt | A gold charm necklace with colorful gemstone pendants... |
| 方法 | FlowAlign (WANAlign2.1) |
| 输出 | `results/pvtt/flowalign_bracelet_to_necklace.mp4` |

### 配置文件

- `config/pvtt/bracelet_to_necklace.yaml`

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `strength` | 0.7 | 编辑强度 |
| `target_guidance_scale` | 19.5 | 目标引导强度 |
| `flag_attnmask` | True | 启用 attention masking (DIFS) |
| `zeta_scale` | 1e-3 | 保持源视频特征的强度 |

## 评估指标

运行完成后，评估以下方面：

1. **产品替换成功率**：手链是否被替换为项链
2. **背景保持**：紫色丝绸背景是否保持
3. **时序一致性**：产品在帧间是否稳定
4. **产品细节**：项链的细节是否清晰

## 预期问题

基于之前的观察，预期会有以下问题：

1. 主体时序一致性差
2. 不完全跟随指令编辑
3. 产品细节丢失
4. 形态迁移困难（手链→项链形态差异大）

这些问题将作为 PVTT 方法需要解决的核心挑战。

## 文件结构

```
baseline/
├── README.md
├── setup_5090.sh          # 环境设置脚本
├── run_pvtt_baseline.sh   # 运行实验脚本
└── flowedit-wan/          # FlowEdit + WAN2.1 代码
    ├── config/
    │   └── pvtt/
    │       └── bracelet_to_necklace.yaml
    ├── videos/
    │   └── bracelet_shot1.mp4  # 测试视频
    └── results/
        └── pvtt/               # 实验结果
```
