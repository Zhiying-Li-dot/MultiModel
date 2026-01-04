# PVTT: Product Video Template Transfer

> 给定一个成功的产品推广视频模板和新产品的图片，生成新产品的推广视频，保持模板的风格、运镜、节奏。

## 任务定义

**输入**：
1. Template Video：一个成功的产品推广视频
2. New Product Image(s)：新产品的图片（1-N张）

**输出**：
- 新产品的推广视频
- 保持模板的视觉风格和动态结构

## 项目结构

```
pvtt/
├── data/                 # 数据集
│   └── samples/          # 样例数据
├── src/                  # 源代码
├── experiments/          # 实验记录
│   ├── README.md         # 实验结果汇总
│   └── baseline/         # Baseline 实验
└── docs/                 # 文档
    ├── research-plan.md  # 研究计划
    ├── literature-review.md  # 文献综述
    └── baseline-design.md    # Baseline 设计
```

## 研究进度

- [x] 调研相关工作
  - [x] DreamSwapV (2025)
  - [x] VideoSwap (CVPR 2024)
  - [x] MotionBooth (NeurIPS 2024)
- [x] 收集模板视频样例
- [x] Baseline 实验 (WANAlign2.1)
  - ✅ 产品替换成功
  - ⚠️ 时序一致性问题
  - ⚠️ 细节清晰度不足
- [ ] 设计 PVTT 技术方案
- [ ] 数据集构建
- [ ] 完整实验
- [ ] 论文写作

## 快速链接

| 文档 | 说明 |
|------|------|
| [研究计划](docs/research-plan.md) | 目标、时间表、阶段任务 |
| [文献综述](docs/literature-review.md) | 论文精读、技术对比 |
| [Baseline 设计](docs/baseline-design.md) | FlowEdit + WAN2.1 |
| [实验结果](experiments/README.md) | 实验配置与结果 |

## 目标会议

CVPR 2027 (Deadline: 2026年11月中旬)
