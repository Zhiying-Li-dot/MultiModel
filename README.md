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
│   ├── templates/        # 模板视频
│   ├── products/         # 产品图片
│   └── pairs/            # 配对数据
├── src/                  # 源代码
│   ├── models/           # 模型定义
│   ├── data/             # 数据处理
│   └── utils/            # 工具函数
├── experiments/          # 实验记录
├── docs/                 # 文档
└── 2026-research-plan.md # 研究计划
```

## 研究进度

- [x] 调研相关工作
  - [x] DreamSwapV (2025)
  - [x] VideoSwap (CVPR 2024)
  - [x] MotionBooth (NeurIPS 2024)
- [ ] 收集模板视频样例
- [ ] 初步实验（IP-Adapter + Wan）
- [ ] 设计技术方案
- [ ] 实现 Baseline
- [ ] 数据集构建
- [ ] 完整实验
- [ ] 论文写作

## 目标会议

CVPR 2027 (Deadline: 2026年11月中旬)

## 相关论文

| 论文 | 会议 | 链接 |
|------|------|------|
| DreamSwapV | Under Review | [arXiv](https://arxiv.org/abs/2508.14465) |
| VideoSwap | CVPR 2024 | [GitHub](https://github.com/showlab/VideoSwap) |
| MotionBooth | NeurIPS 2024 | [GitHub](https://github.com/jianzongwu/MotionBooth) |
