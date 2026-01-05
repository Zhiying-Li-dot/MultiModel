# 2026 研究计划：Product Video Template Transfer (PVTT)

## 2026 目标

**发表一篇 CVPR 级别的一作论文**

目标会议：CVPR 2027（Deadline: 2026年11月中旬）

## 资源评估

| 资源 | 状态 |
|------|------|
| 时间 | 每周 10 小时起步，可调整 |
| GPU | H800 Superpod，8卡可用 |
| 数据 | 可自行收集，京东合作在洽谈中 |
| 方向 | Video editing，与创业方向吻合 |

## 核心洞察

> 电商场景需要的是编辑，不是生成。商家有素材，难点是把素材变成高品质视频。

现有 Training-Free video editing 方法的问题：
1. 主体时序一致性差
2. 指令遵循差

## 研究任务定义

### Product Video Template Transfer (PVTT)

**任务描述**：给定一个成功的产品推广视频模板和新产品的图片，生成新产品的推广视频，保持模板的风格、运镜、节奏。

**输入**：
1. Template Video：一个成功的产品推广视频
2. New Product Image(s)：新产品的图片（1-N张）

**输出**：
- 新产品的推广视频
- 保持模板的视觉风格和动态结构

**核心挑战**：
1. 保持模板的视觉风格和动态结构
2. 产品形态可能完全不同（如：手机 → 化妆品）
3. 产品细节的保真度

## 研究路线

选择 **Task Paper** 路线：定义新任务 + Baseline 方法 + 小规模数据集

**优势**：
1. 定义任务，别人在你的赛道上比赛
2. Motivation 强，真实商业需求
3. 不需要 SOTA 方法，只需要合理的 baseline
4. 数据集不需要太大（100-200 个样本）
5. 可以持续发 follow-up

## 时间表

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| **1. 调研 & 定义** | 1-2月 | 相关工作调研、细化任务定义、确定评估指标 | 任务定义文档、Related Work 初稿 |
| **2. 数据收集** | 2-4月 | 收集模板视频、产品图、标注 | 100+ 三元组数据集 |
| **3. Baseline 开发** | 4-7月 | 搭建 pipeline、尝试不同方法 | 可运行的 baseline 系统 |
| **4. 实验 & 迭代** | 7-9月 | 完整实验、消融实验、对比实验 | 实验结果、可视化 |
| **5. 写作** | 9-10月 | 写论文、做图表 | 论文初稿 |
| **6. 打磨 & 投稿** | 10-11月 | 润色、补实验、提交 | 最终投稿 |

## 阶段 1 详细计划（2026年1-2月）

### 目标
完成调研，确认没有高度重合的现有工作

### 具体任务

**1. 调研相关工作（约10小时）**
- Video editing 方法：TokenFlow, FateZero, InsV2V, etc.
- Subject-driven generation：DreamBooth, IP-Adapter
- Reference-based video generation
- 确认 PVTT 任务的 novelty

**2. 细化任务定义（约5小时）**
- 输入输出格式
- 评估指标：保真度、风格一致性、时序平滑度、用户研究
- 数据集格式定义

**3. 初步验证（约5小时）**
- 用现有方法（如 IP-Adapter + Wan）跑几个例子
- 分析现有方法的失败模式

### Action Items

- [x] 文献调研：搜索 "product video generation"、"video template"、"subject-driven video editing"
  - 发现 E-CommerceVideo (淘宝/NeurIPS 2025)，但任务不同：I2V generation vs template transfer
  - PVTT 任务有 novelty ✅
- [ ] 收集 2-3 个典型的模板视频样例
- [x] 用现有方法做初步实验（Baseline 完成）
- [x] 确定 Baseline 方法：**FlowEdit**（ICLR 2025 Best Paper，简单直观）

## 相关文档

- [文献综述](./literature-review.md)
- [Baseline 设计](./baseline-design.md)
- [实验结果](../experiments/README.md)
