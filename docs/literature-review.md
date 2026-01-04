# 文献综述

## 一、核心学术论文

### 1. DreamVideo (CVPR 2024)
- **任务**：给定几张主体图片 + 几个动作视频 → 生成自定义主体执行目标动作的视频
- **方法**：两阶段解耦（subject learning + motion learning），使用 identity adapter 和 motion adapter
- **局限**：需要动作视频作为输入，不是模板迁移
- **链接**：[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_DreamVideo_Composing_Your_Dream_Videos_with_Customized_Subject_and_Motion_CVPR_2024_paper.html) | [Project](https://dreamvideo-t2v.github.io/)

### 2. VideoSwap (CVPR 2024)
- **任务**：将源视频中的主体替换为目标主体，保留背景和运动轨迹
- **方法**：基于语义点对应（semantic point correspondence），只需少量关键点对齐
- **局限**：需要交互式标注语义点，主要关注主体形状变化
- **链接**：[GitHub](https://github.com/showlab/VideoSwap)

### 3. MotionBooth (NeurIPS 2024 Spotlight)
- **任务**：自定义主体 + 控制主体运动和相机运动
- **方法**：训练时使用 subject region loss，推理时使用 cross-attention 操控 + latent shift 控制相机
- **特点**：首个同时支持主体自定义、主体运动、相机运动的框架
- **链接**：[Paper](https://arxiv.org/abs/2406.17758) | [GitHub](https://github.com/jianzongwu/MotionBooth)

### 4. DreamSwapV (2025, Under Review)
- **任务**：给定 mask + 参考图片 → 替换视频中任意主体
- **方法**：端到端框架，condition fusion module + adaptive mask strategy
- **特点**：subject-agnostic，不限于特定领域（如人脸、人体）
- **链接**：[Paper](https://arxiv.org/html/2508.14465)

### 5. I2VEdit (SIGGRAPH Asia 2024)
- **任务**：编辑第一帧 → 传播到整个视频
- **方法**：基于 Image-to-Video diffusion，保持时序一致性
- **链接**：[Paper](https://dl.acm.org/doi/10.1145/3680528.3687656)

---

## 二、与 PVTT 任务对比

| 现有工作 | 输入 | PVTT 任务 | 差异 |
|---------|------|-----------|------|
| DreamVideo | 主体图 + 动作视频 | 模板视频 + 产品图 | PVTT 的模板包含完整风格/运镜 |
| VideoSwap | 源视频 + 目标主体 | 模板视频 + 产品图 | PVTT 需要保持模板"风格"而非仅替换主体 |
| MotionBooth | 主体图 + 运动控制 | 模板视频 + 产品图 | PVTT 的运动来自模板，不需要显式控制 |
| DreamSwapV | mask + 参考图 | 模板视频 + 产品图 | 最接近，但 PVTT 强调"模板风格迁移" |

---

## 三、论文精读笔记

### DreamSwapV (2025, Under Review)

#### 任务定义
- **输入**：Source Video + User-specified Mask + Reference Image
- **输出**：将 mask 区域的主体替换为 reference 中的主体
- **核心创新**：Subject-agnostic，端到端框架

#### 方法架构

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

#### 可借鉴技术
- ⭐⭐⭐ Condition Fusion Module 设计
- ⭐⭐⭐ Reference 处理方式（frame-level concat + KV 隔离）
- ⭐⭐⭐ Benchmark 设计思路

---

### VideoSwap (CVPR 2024)

#### 核心洞察
> 只需要**少量语义点**就足以对齐主体运动轨迹并修改形状

#### 方法架构

**基础模型**：Latent Diffusion Model + AnimateDiff + Co-Tracker + LNA

**三种对应场景**：
| 类型 | 场景 | 处理方式 |
|------|------|---------|
| One-to-One | 源和目标语义点对齐 | 直接使用源点序列 |
| Partial | 特征不匹配 | 用户删除多余点 |
| Shape Morphing | 形状差异大 | 用户拖动点 |

**关键技术**：
- Point Patch Loss：只在点周围局部 patch 计算 loss
- Semantic-Enhanced Schedule：T_min = T/2

#### 可借鉴技术
- ⭐⭐⭐ Sparse motion features 思路
- ⭐⭐ Point Patch Loss（局部 loss 防止 leakage）

---

### MotionBooth (NeurIPS 2024 Spotlight)

#### 核心创新
> 首个同时支持**自定义主体 + 主体运动控制 + 相机运动控制**的框架

#### 方法架构

**训练阶段 Loss**：
| Loss | 作用 | 机制 |
|------|------|------|
| Subject Region Loss | 防止背景过拟合 | 只在主体 mask 区域计算 loss |
| Video Preservation Loss | 保持视频生成能力 | 联合训练随机视频 |
| STCA Loss | 建立 token 与位置绑定 | BCE loss 引导 attention |

**推理阶段**：
| 控制类型 | 输入 | 机制 |
|---------|------|------|
| 主体运动 | Bbox 序列 | Cross-attention editing |
| 相机运动 | 速度参数 | Latent shift |

#### 可借鉴技术
- ⭐⭐⭐ Subject Region Loss
- ⭐⭐⭐ STCA Loss
- ⭐⭐⭐ Video Preservation Loss

---

## 四、三篇论文总结对比

| 维度 | DreamSwapV | VideoSwap | MotionBooth |
|------|-----------|-----------|-------------|
| 会议 | Under Review | CVPR 2024 | NeurIPS 2024 |
| 核心思路 | Video Inpainting | 语义点对应 | Fine-tune + Attention 操控 |
| 需要 Mask | ✅ | ❌ | ❌（但需要 bbox） |
| 需要交互 | ❌ | ✅ | ✅ |
| 对 PVTT 最有价值 | Condition Fusion | Sparse Features | Region Loss + STCA |

---

## 五、PVTT 技术路线初步设想

1. **输入处理**
   - 模板视频 → 提取运动信息（bbox 序列 / attention pattern / 光流）
   - 产品图片 → 编码产品外观

2. **模型设计**
   - 借鉴 DreamSwapV 的 condition fusion
   - 借鉴 MotionBooth 的 STCA loss

3. **训练策略**
   - 借鉴 MotionBooth 的 region loss
   - 借鉴 MotionBooth 的 video preservation loss

4. **推理**
   - 用模板的运动信息引导生成
   - 产品出现在模板中产品对应的位置
