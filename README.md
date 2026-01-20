# PVTT: Product Video Template Transfer

> 给定一个成功的产品推广视频模板和新产品的图片，生成新产品的推广视频，保持模板的风格、运镜、节奏。

## 任务定义

**输入**：
1. Template Video：一个成功的产品推广视频
2. New Product Image(s)：新产品的图片（1-N张）

**输出**：
- 新产品的推广视频
- 保持模板的视觉风格和动态结构

## 当前最佳方案

**Flux.2 + TI2V 两阶段组合方法**

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Flux.2 图像编辑                                         │
│   源视频首帧 + 产品图片 → 目标首帧（产品替换）                      │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: Wan2.2 TI2V                                            │
│   目标首帧 + prompt → 目标视频                                    │
└─────────────────────────────────────────────────────────────────┘
```

详细运行方法见 **[实验运行指南](docs/running-experiments.md)**。

## 项目结构

```
pvtt/
├── baseline/
│   ├── flowedit-wan2.1/          # Wan2.1 T2V-1.3B (FlowEdit/FlowAlign)
│   ├── flowedit-wan2.2/          # Wan2.2 TI2V-5B (FlowEdit/FlowAlign)
│   └── compositional-flux-ti2v/  # ⭐ Flux.2 + TI2V 组合方法
│       └── scripts/
│           └── ti2v_rfsolver.py  # ⭐ RF-Solver Inversion + TI2V
├── data/
│   ├── samples/                  # 样例数据
│   └── pvtt-benchmark/           # ⭐ 标准化测试用例
│       └── cases/{case_name}/    # source_video.mp4, target_frame1.png, config.yaml
├── experiments/
│   ├── README.md                 # 实验索引
│   ├── logs/                     # ⭐ 实验日志 (主题_日期.md)
│   └── results/                  # 实验输出
├── docs/
│   ├── running-experiments.md    # ⭐ 实验运行指南
│   ├── design/                   # 技术方案
│   ├── reports/                  # 周报
│   └── research-plan.md          # 研究计划
└── scripts/                      # 工具脚本
```

## 研究进度

- [x] 调研相关工作 (DreamSwapV, VideoSwap, MotionBooth)
- [x] Baseline 实验
  - [x] FlowEdit/FlowAlign × Wan2.1/Wan2.2 对比
  - [x] RefDrop 图像条件方案（失败：与 self-attention 编辑机制冲突）
  - [x] **Flux.2 + TI2V 组合方法**（成功：产品替换效果好）
- [x] TI2V + FlowEdit 实验
  - [x] 修复 CFG bug（必须用空字符串作 negative prompt）
  - [x] 验证 ti2v_flowedit.py 与 flowalign_t2v.py 像素级一致
  - [x] 分析 Inversion-Free 根本问题
- [x] 设计 Flow Matching Inversion + TI2V 方案
- [x] 实现 RF-Solver Inversion + TI2V
  - [x] shift 参数消融：**shift=0.5 最佳**
  - [x] 帧数消融：发现 **33-49 帧是"坏区间"**，17-25 帧和 81 帧效果好
- [ ] 数据集构建
- [ ] 论文写作

## 关键实验结论

| 结论 | 说明 |
|------|------|
| Flux.2 + TI2V 目前最佳 | 产品替换效果好，首帧质量高 |
| RF-Solver shift=0.5 最佳 | 比默认 shift=5.0 效果好很多 |
| 33-49 帧是"坏区间" | 17-25 帧和 81 帧效果好，33-49 帧质量差 |
| std ≠ 视觉质量 | std 接近 1.0 不代表视觉质量好 |
| FlowAlign > FlowEdit | 3-branch 编辑效果强于 2-branch |
| TI2V + FlowEdit 图像条件失败 | Inversion-Free 导致后续帧退化 |

## 技术方案

| 方案 | 状态 | 说明 |
|------|------|------|
| [Flux.2 + TI2V](baseline/compositional-flux-ti2v/) | ✅ 完成 | 当前最佳，两阶段组合 |
| [RF-Solver Inversion + TI2V](baseline/compositional-flux-ti2v/scripts/ti2v_rfsolver.py) | ✅ 完成 | 二阶 inversion，shift=0.5 |

## Baseline 方法

| 方法 | 论文 | 说明 |
|------|------|------|
| FlowEdit | [arXiv:2412.08629](https://arxiv.org/abs/2412.08629) | Inversion-Free 视频编辑 |
| FlowAlign | - | FlowEdit + zeta 正则化 |
| RF-Solver | [arXiv:2411.04746](https://arxiv.org/abs/2411.04746) | Flow Matching Inversion |

## 快速链接

| 文档 | 说明 |
|------|------|
| [实验运行指南](docs/running-experiments.md) | 如何运行各个脚本 |
| [Flow Matching Inversion 方案](docs/design/rf-inversion-ti2v.md) | 技术设计文档 |
| [周报](docs/reports/) | 每周实验进展 |
| [实验结果](experiments/README.md) | 实验配置与结果 |

## 目标会议

CVPR 2027 (Deadline: 2026年11月中旬)
