# 2026 研究计划：Product Video Template Transfer (PVTT)

## 背景

- 2019年开始港科大博士
- 主要时间在创业，没有固定导师指导
- 过去涉猎：人体姿态估计、lip sync、video editing
- 2024年 lip sync 工作投稿 CVPR 被拒（原因：方法不够 novel，用 GAN 偏保守）
- 目前在做 video editing 方向，带学生做 Training-Free 和 DMD 两个子工作

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

- [ ] 文献调研：搜索 "product video generation"、"video template"、"subject-driven video editing"
- [ ] 收集 2-3 个典型的模板视频样例
- [ ] 用现有方法做初步实验

## 相关工作（2026年1月4日调研）

### 一、核心学术论文

#### 1. DreamVideo (CVPR 2024)
- **任务**：给定几张主体图片 + 几个动作视频 → 生成自定义主体执行目标动作的视频
- **方法**：两阶段解耦（subject learning + motion learning），使用 identity adapter 和 motion adapter
- **局限**：需要动作视频作为输入，不是模板迁移
- **链接**：[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_DreamVideo_Composing_Your_Dream_Videos_with_Customized_Subject_and_Motion_CVPR_2024_paper.html) | [Project](https://dreamvideo-t2v.github.io/)

#### 2. VideoSwap (CVPR 2024)
- **任务**：将源视频中的主体替换为目标主体，保留背景和运动轨迹
- **方法**：基于语义点对应（semantic point correspondence），只需少量关键点对齐
- **局限**：需要交互式标注语义点，主要关注主体形状变化
- **链接**：[GitHub](https://github.com/showlab/VideoSwap)

#### 3. MotionBooth (NeurIPS 2024 Spotlight)
- **任务**：自定义主体 + 控制主体运动和相机运动
- **方法**：训练时使用 subject region loss，推理时使用 cross-attention 操控 + latent shift 控制相机
- **特点**：首个同时支持主体自定义、主体运动、相机运动的框架
- **链接**：[Paper](https://arxiv.org/abs/2406.17758) | [GitHub](https://github.com/jianzongwu/MotionBooth)

#### 4. DreamSwapV (2025, Under Review)
- **任务**：给定 mask + 参考图片 → 替换视频中任意主体
- **方法**：端到端框架，condition fusion module + adaptive mask strategy
- **特点**：subject-agnostic，不限于特定领域（如人脸、人体）
- **链接**：[Paper](https://arxiv.org/html/2508.14465) | [OpenReview](https://openreview.net/forum?id=xH0pSRWbFi)

#### 5. I2VEdit (SIGGRAPH Asia 2024)
- **任务**：编辑第一帧 → 传播到整个视频
- **方法**：基于 Image-to-Video diffusion，保持时序一致性
- **链接**：[Paper](https://dl.acm.org/doi/10.1145/3680528.3687656)

#### 6. I2V-Adapter (SIGGRAPH 2024)
- **任务**：将图像作为条件输入，引导视频生成
- **方法**：修改 cross-attention layers，保留 temporal attention
- **链接**：[Paper](https://dl.acm.org/doi/10.1145/3641519.3657407)

### 二、与 PVTT 任务对比

| 现有工作 | 输入 | PVTT 任务 | 差异 |
|---------|------|-----------|------|
| DreamVideo | 主体图 + 动作视频 | 模板视频 + 产品图 | PVTT 的模板包含完整风格/运镜 |
| VideoSwap | 源视频 + 目标主体 | 模板视频 + 产品图 | PVTT 需要保持模板"风格"而非仅替换主体 |
| MotionBooth | 主体图 + 运动控制 | 模板视频 + 产品图 | PVTT 的运动来自模板，不需要显式控制 |
| DreamSwapV | mask + 参考图 | 模板视频 + 产品图 | 最接近，但 PVTT 强调"模板风格迁移" |

### 三、关键发现

#### Novelty 空间存在

**现有工作的共同假设**：主体替换/自定义时，目标是保持"运动轨迹"或"语义对应"

**PVTT 的独特视角**：
- 保持的是**模板的整体风格**（运镜、剪辑节奏、视觉风格）
- 产品可能形态完全不同（手机 → 化妆品）
- 商业驱动的明确 use case

#### 电商视频生成的学术空白

商业工具很多（Creatify、Zeely、Pippit、Buñu），但**学术论文几乎没有专门针对电商产品视频的工作**。

### 四、潜在技术路线

| 方案 | 思路 | 优劣 |
|------|------|------|
| **A. VideoSwap 改进** | 用语义点对应做主体替换 | 需要标注，不够端到端 |
| **B. I2VEdit 路线** | 编辑第一帧 → 传播 | 简单，但风格迁移能力有限 |
| **C. IP-Adapter + Wan** | 用产品图作为 reference condition | 灵活，但需要设计如何融入模板信息 |
| **D. Motion + Subject 双分支** | 分别建模模板运动和产品外观 | 类似 DreamVideo，但用模板视频代替运动视频 |

### 五、需要精读的论文

- [x] VideoSwap (CVPR 2024) - 已完成详细分析
- [x] DreamSwapV (2025) - 已完成详细分析
- [x] MotionBooth (NeurIPS 2024) - 已完成详细分析

---

## 论文精读笔记

### DreamSwapV (2025, Under Review)

#### 基本信息
- **标题**：DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing
- **链接**：[arXiv](https://arxiv.org/abs/2508.14465)

#### 任务定义
- **输入**：Source Video + User-specified Mask + Reference Image
- **输出**：将 mask 区域的主体替换为 reference 中的主体，保持与环境的自然交互
- **核心创新**：Subject-agnostic，端到端框架

#### 方法架构

**整体思路**：将 subject swapping 建模为 video inpainting 任务

**Condition Fusion Module**：
| 条件类型 | 符号 | 作用 |
|---------|------|------|
| Binary Mask | M^s | 指定替换区域 |
| Pose/3D Hand | P | 运动引导（可选） |
| Masked Video | A^s | 背景信息 |
| Reference Image | r^s | 目标外观 |

**Reference 处理**：
- Frame-level concatenation（扩展 token 长度）
- Reference 在 self-attention 中只 attend to 自己（KV cache 隔离）

**Adaptive Mask Strategy**：
- Adaptive Grid Sizing：网格分辨率与主体尺度成反比
- Shape Augmentation：训练时在 mask 边缘添加随机几何形状
- Bounding Box Aug：30% 概率使用 bounding box

#### 训练策略

**两阶段训练**：
| 阶段 | 训练内容 | 数据 |
|------|---------|------|
| Phase 1 | 只训练 self-attention layers | HumanVID 衍生（8,160 视频） |
| Phase 2 | 全模型微调 | AnyInsertion + Subject200K + AnchorCrafter-400 |

**Loss 函数**：`L = (1 - M^s) ⊙ L_pt + λ * L_rw`

#### 实验结果
| Method | VBench Avg | User Study Avg |
|--------|-----------|----------------|
| **DreamSwapV** | **80.44%** | **71.41%** |
| Kling 1.6 | 79.79% | 68.63% |
| HunyuanCustom | 78.17% | 68.61% |

#### 与 PVTT 对比

| 维度 | DreamSwapV | PVTT |
|------|-----------|------|
| 输入 | 源视频 + Mask + Reference | 模板视频 + 产品图 |
| 需要 Mask | ✅ 需要 | ❌ 不需要 |
| 目标 | 替换任意主体 | 用新产品"重演"模板 |
| 保持什么 | 背景、主体与环境交互 | 模板的风格、运镜、节奏 |

#### 可借鉴技术
- ⭐⭐⭐ Condition Fusion Module 设计
- ⭐⭐⭐ Reference 处理方式（frame-level concat + KV 隔离）
- ⭐⭐⭐ Benchmark 设计思路
- ⭐⭐ 两阶段训练策略

#### PVTT 的机会点
1. 不需要 mask 的端到端方案
2. 强调模板风格迁移（DreamSwapV 未 explicitly 建模）
3. 电商场景的独特数据

---

### VideoSwap (CVPR 2024)

#### 基本信息
- **标题**：VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence
- **机构**：ShowLab (NUS) + GenAI (Meta)
- **链接**：[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Gu_VideoSwap_Customized_Video_Subject_Swapping_with_Interactive_Semantic_Point_Correspondence_CVPR_2024_paper.html) | [GitHub](https://github.com/showlab/VideoSwap)

#### 核心洞察
> 只需要**少量语义点**就足以对齐主体运动轨迹并修改形状，不需要 dense correspondence

#### 方法架构

**基础模型**：Latent Diffusion Model + AnimateDiff motion layer + Co-Tracker + Layered Neural Atlas (LNA)

**三种对应场景**：
| 类型 | 场景 | 处理方式 |
|------|------|---------|
| One-to-One | 源和目标语义点对齐（狗→猫） | 直接使用源点序列作为运动引导 |
| Partial | 特征不匹配（飞机→直升机） | 用户删除多余点 |
| Shape Morphing | 对应存在但形状差异大 | 用户在关键帧拖动点 |

**语义点注册**：
- 用 Co-Tracker 追踪 8 帧语义点
- MLP 投影点 embedding → sparse motion features
- 100 iterations 优化，Adam lr=5e-5

**关键技术**：
- Point Patch Loss：只在点周围局部 patch 计算 loss，减少 structure leakage
- Semantic-Enhanced Schedule：T_min = T/2，高 timestep 学语义

#### 实验结果
| 指标 | VideoSwap |
|------|-----------|
| CLIP Text Alignment | 26.87 (最高) |
| Temporal Consistency | 95.93 (最高) |
| Human Preference vs DDIM | 73% |

#### 局限性
- 点追踪在自遮挡、大视角变化时失败
- LNA 无法表示 3D 旋转
- 计算成本高：4分钟预处理 + 50秒/编辑
- **需要用户交互标注**

#### 与 PVTT 对比

| 维度 | VideoSwap | PVTT |
|------|-----------|------|
| 输入 | 源视频 + 语义点 + 目标主体图 | 模板视频 + 产品图 |
| 需要交互 | ✅ 需要标注/拖动语义点 | ❌ 希望端到端 |
| 保持什么 | 背景 + 运动轨迹 | 模板风格、运镜、节奏 |

#### 可借鉴技术
- ⭐⭐⭐ Sparse motion features 思路
- ⭐⭐ Point Patch Loss（局部 loss 防止 leakage）
- ⭐⭐ Semantic-Enhanced Schedule
- ⭐ LNA canonical space（计算成本太高）

#### 对 PVTT 的启发
1. 能否**自动**检测产品的关键语义点？
2. 能否从模板提取"运动骨架"，然后用新产品填充？
3. 借鉴局部 loss 设计，在产品区域加强学习

---

### MotionBooth (NeurIPS 2024 Spotlight)

#### 基本信息
- **标题**：MotionBooth: Motion-Aware Customized Text-to-Video Generation
- **机构**：北大、NTU、上海AI Lab、浙大、上交
- **链接**：[Paper](https://arxiv.org/abs/2406.17758) | [GitHub](https://github.com/jianzongwu/MotionBooth)

#### 核心创新
> 首个同时支持**自定义主体 + 主体运动控制 + 相机运动控制**的框架

#### 方法架构

**训练阶段**：Fine-tune T2V 模型（Zeroscope/LaVie），使用三种 loss

| Loss | 作用 | 机制 |
|------|------|------|
| Subject Region Loss | 防止背景过拟合 | 只在主体 mask 区域计算重建 loss |
| Video Preservation Loss | 保持视频生成能力 | 联合训练随机视频（Panda-70M） |
| STCA Loss | 建立 token 与位置绑定 | BCE loss 引导 [V] token attention 落在主体区域 |

**推理阶段**：Training-Free 运动控制

| 控制类型 | 输入 | 机制 |
|---------|------|------|
| 主体运动 | Bbox 序列 | Cross-attention editing，在 bbox 内放大 [V] attention |
| 相机运动 | 速度参数 | Latent shift，中间步骤位移 latent |

#### 训练配置
- 300 steps，~20 分钟（单卡 A100）
- 80GB GPU 内存（24帧视频）

#### 实验结果

| 方法 | R-CLIP ↑ | R-DINO ↑ | Temporal ↑ |
|------|----------|----------|------------|
| DreamBooth | 0.762 | 0.524 | 0.956 |
| DreamVideo | 0.789 | 0.561 | 0.959 |
| **MotionBooth** | **0.801** | **0.583** | **0.968** |

相机控制 Flow Error：0.190-0.296（vs AnimateDiff 1.683, CameraCtrl 0.807）

#### 局限性
- 多主体困难
- 难以生成与主体语义不符的动作
- 依赖基础模型能力

#### 与 PVTT 对比

| 维度 | MotionBooth | PVTT |
|------|------------|------|
| 输入 | 主体图片 + 运动控制信号 | 模板视频 + 产品图 |
| 运动来源 | 用户指定 bbox/相机速度 | 从模板视频提取 |
| 目标 | 生成主体执行指定动作 | 用新产品"重演"模板 |

#### 可借鉴技术
- ⭐⭐⭐ Subject Region Loss（产品区域加强学习）
- ⭐⭐⭐ STCA Loss（建立 token 与位置绑定）
- ⭐⭐⭐ Video Preservation Loss（保持视频能力）
- ⭐⭐ Cross-Attention Editing（引导产品位置）

#### 对 PVTT 的启发
1. 从模板视频自动提取 bbox 序列或运动表示
2. Region Loss 在产品区域加强学习，背景保持模板原样
3. 用模板的 attention pattern 引导新产品生成
4. 两阶段：先学产品外观，再用模板运动引导

---

## 三篇论文总结对比

| 维度 | DreamSwapV | VideoSwap | MotionBooth |
|------|-----------|-----------|-------------|
| 会议 | Under Review | CVPR 2024 | NeurIPS 2024 |
| 核心思路 | Video Inpainting | 语义点对应 | Fine-tune + Attention 操控 |
| 需要 Mask | ✅ | ❌ | ❌（但需要 bbox） |
| 需要交互 | ❌ | ✅ | ✅（需要指定运动） |
| 运动来源 | 保持原视频 | 保持原视频 | 用户指定 |
| 对 PVTT 最有价值 | Condition Fusion | Sparse Features | Region Loss + STCA |

### PVTT 技术路线初步设想

基于三篇论文的分析，可能的技术路线：

1. **输入处理**
   - 模板视频 → 提取运动信息（bbox 序列 / attention pattern / 光流）
   - 产品图片 → 编码产品外观

2. **模型设计**
   - 借鉴 DreamSwapV 的 condition fusion（融合模板 + 产品条件）
   - 借鉴 MotionBooth 的 STCA loss（绑定产品 token 与位置）

3. **训练策略**
   - 借鉴 MotionBooth 的 region loss（产品区域加强）
   - 借鉴 MotionBooth 的 video preservation loss（保持视频能力）

4. **推理**
   - 用模板的运动信息引导生成
   - 产品出现在模板中产品对应的位置

---

## 备注

- 该计划基于 2026年1月4日的讨论制定
- 需要根据调研结果和实际进展动态调整
