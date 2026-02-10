# AIGC 核心技术详解：Diffusion Model & Flow Matching

> 基于 [Zhiying-Li-dot/MultiModel](https://github.com/Zhiying-Li-dot/MultiModel) 仓库整理，补充详细公式推导与参数解释。

---

## 目录

- [一、Diffusion Model（扩散模型）](#一diffusion-model扩散模型)
  - [1.1 直觉理解](#11-直觉理解)
  - [1.2 前向过程（加噪）](#12-前向过程加噪)
  - [1.3 反向过程（去噪）](#13-反向过程去噪)
  - [1.4 训练目标](#14-训练目标)
  - [1.5 Classifier-Free Guidance（CFG）](#15-classifier-free-guidancecfg)
  - [1.6 潜空间扩散（Latent Diffusion / Stable Diffusion）](#16-潜空间扩散latent-diffusion--stable-diffusion)
  - [1.7 采样器加速](#17-采样器加速)
  - [1.8 高级应用：ControlNet 与 LoRA](#18-高级应用controlnet-与-lora)
- [二、Flow Matching](#二flow-matching)
  - [2.1 动机：为什么需要 Flow Matching？](#21-动机为什么需要-flow-matching)
  - [2.2 连续归一化流（CNF）回顾](#22-连续归一化流cnf回顾)
  - [2.3 条件 Flow Matching（CFM）](#23-条件-flow-matchingcfm)
  - [2.4 最优传输 CFM（OT-CFM）](#24-最优传输-cfmot-cfm)
  - [2.5 Rectified Flow](#25-rectified-flow)
  - [2.6 采样过程](#26-采样过程)
- [三、Diffusion vs Flow Matching 全面对比](#三diffusion-vs-flow-matching-全面对比)
- [四、采样器统一视角](#四采样器统一视角)
- [五、推荐学习路径](#五推荐学习路径)

---

## 一、Diffusion Model（扩散模型）

### 1.1 直觉理解

扩散模型的核心思想可以用一句话概括：

> **先把数据「打碎」成纯噪声，再学会把噪声「还原」成数据。**

类比：一滴墨水滴入清水，逐渐扩散变均匀（前向过程）；如果我们能学会这个扩散的逆过程，就能从均匀的墨水中还原出那滴墨水（反向过程）。

```
前向过程（已知，不需学习）：
  x_0 (清晰图像) → x_1 → x_2 → ... → x_T (纯高斯噪声)

反向过程（神经网络学习）：
  x_T (纯噪声) → x_{T-1} → ... → x_1 → x_0 (生成图像)
```

---

### 1.2 前向过程（加噪）

#### 1.2.1 逐步加噪

在每一步 $t$，向数据添加少量高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{1 - \beta_t} \cdot x_{t-1},\ \beta_t \cdot \mathbf{I}\right)
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $x_t$ | 第 $t$ 步的含噪数据 | 随 $t$ 增大，噪声越多，信号越弱 |
| $x_{t-1}$ | 上一步的数据 | 比 $x_t$ 更「干净」 |
| $\beta_t$ | **噪声调度系数**（variance schedule） | 控制每一步加多少噪声，通常从 $\beta_1 = 0.0001$ 线性增长到 $\beta_T = 0.02$ |
| $\sqrt{1 - \beta_t}$ | 信号保留系数 | $\beta_t$ 越大 → 信号衰减越快 |
| $\mathbf{I}$ | 单位矩阵 | 表示各维度独立加噪 |
| $\mathcal{N}(\mu, \sigma^2)$ | 高斯分布 | 均值 $\mu$、方差 $\sigma^2$ |

#### 1.2.2 一步加噪公式（核心）

定义辅助变量：

$$
\alpha_t = 1 - \beta_t \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \alpha_1 \cdot \alpha_2 \cdots \alpha_t
$$

则可以从原始数据 $x_0$ **一步直达**任意时刻 $t$：

$$
\boxed{x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \boldsymbol{\epsilon}}
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $x_0$ | **原始干净数据**（如一张真实图片） | 训练数据集中的样本 |
| $x_t$ | 第 $t$ 步的加噪结果 | 介于干净数据与纯噪声之间 |
| $\alpha_t = 1 - \beta_t$ | 单步信号保留率 | 接近 1，表示每步只衰减一点 |
| $\bar{\alpha}_t$ | **累积信号保留率**（所有步的乘积） | 随 $t$ 增大而递减，$\bar{\alpha}_T \approx 0$ |
| $\sqrt{\bar{\alpha}_t}$ | 信号分量系数 | 控制原始数据 $x_0$ 还剩多少 |
| $\sqrt{1 - \bar{\alpha}_t}$ | 噪声分量系数 | 控制噪声 $\epsilon$ 占多大比例 |
| $\boldsymbol{\epsilon}$ | **标准高斯噪声** | 从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 随机采样 |

**直觉解读：**

- 当 $t = 0$ 时：$\bar{\alpha}_0 = 1$，$x_0 = 1 \cdot x_0 + 0 \cdot \epsilon = x_0$（纯信号）
- 当 $t = T$ 时：$\bar{\alpha}_T \approx 0$，$x_T \approx 0 \cdot x_0 + 1 \cdot \epsilon = \epsilon$（纯噪声）
- 中间时刻：信号与噪声按比例混合

#### 1.2.3 推导过程

从逐步递推到一步公式：

$$
x_1 = \sqrt{\alpha_1} \cdot x_0 + \sqrt{1 - \alpha_1} \cdot \epsilon_1
$$

$$
\begin{aligned}
x_2 &= \sqrt{\alpha_2} \cdot x_1 + \sqrt{1 - \alpha_2} \cdot \epsilon_2 \\
    &= \sqrt{\alpha_2} \left[\sqrt{\alpha_1} \cdot x_0 + \sqrt{1 - \alpha_1} \cdot \epsilon_1\right] + \sqrt{1 - \alpha_2} \cdot \epsilon_2 \\
    &= \underbrace{\sqrt{\alpha_1 \alpha_2}}_{\sqrt{\bar{\alpha}_2}} \cdot x_0 + \underbrace{\sqrt{\alpha_2(1 - \alpha_1)} \cdot \epsilon_1 + \sqrt{1 - \alpha_2} \cdot \epsilon_2}_{\text{两个独立高斯之和}}
\end{aligned}
$$

利用**高斯噪声的可加性**——两个独立高斯之和仍是高斯，方差相加：

$$
\text{合并后方差} = \alpha_2(1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1 \alpha_2 = 1 - \bar{\alpha}_2
$$

因此：

$$
x_2 = \sqrt{\bar{\alpha}_2} \cdot x_0 + \sqrt{1 - \bar{\alpha}_2} \cdot \boldsymbol{\epsilon}
$$

递推至第 $t$ 步，即得到一步加噪公式。

---

### 1.3 反向过程（去噪）

反向过程的目标是学习条件分布 $p_\theta(x_{t-1} \mid x_t)$，将含噪数据逐步还原：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\ \boldsymbol{\mu}_\theta(x_t, t),\ \sigma_t^2 \cdot \mathbf{I}\right)
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $p_\theta$ | 神经网络参数化的分布 | $\theta$ 是网络的可学习参数 |
| $\boldsymbol{\mu}_\theta(x_t, t)$ | 网络预测的去噪均值 | 给定 $x_t$ 和时刻 $t$，预测 $x_{t-1}$ 的期望 |
| $\sigma_t^2$ | 预设方差 | 通常取 $\beta_t$ 或 $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ |

通过贝叶斯公式，可以推导出后验均值的解析形式。由于网络选择**预测噪声** $\boldsymbol{\epsilon}_\theta$，最终的**核心去噪公式**为：

$$
\boxed{x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \boldsymbol{\epsilon}_\theta(x_t, t) \right) + \sigma_t \cdot \mathbf{z}}
$$

$$
\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\quad \text{当 } t = 1 \text{ 时 } \mathbf{z} = \mathbf{0}
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $x_t$ | 当前含噪数据 | 输入给网络 |
| $x_{t-1}$ | 去噪一步后的结果 | 比 $x_t$ 更接近原始数据 |
| $\frac{1}{\sqrt{\alpha_t}}$ | 信号恢复系数 | 补偿前向过程中的信号衰减 |
| $\frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}$ | 噪声移除系数 | 根据当前时刻的噪声水平，决定移除多少预测噪声 |
| $\boldsymbol{\epsilon}_\theta(x_t, t)$ | **网络预测的噪声** | UNet 等网络的输出，估计 $x_t$ 中包含的噪声 |
| $\sigma_t$ | 随机扰动强度 | 为去噪过程引入随机性，增加生成多样性 |
| $\mathbf{z}$ | 随机噪声项 | 标准高斯采样；最后一步 ($t=1$) 不加噪声 |

**关键洞察**：网络不直接预测 $x_{t-1}$，而是**预测噪声 $\boldsymbol{\epsilon}_\theta$**。知道了噪声，就能从 $x_t$ 中「减去」噪声来恢复信号。

---

### 1.4 训练目标

训练损失非常简洁——一个**噪声预测的均方误差**：

$$
\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \boldsymbol{\epsilon},\, t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t) \right\|^2 \right]}
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbb{E}_{x_0, \boldsymbol{\epsilon}, t}$ | 期望，对三个随机变量取平均 | $x_0$ 从数据集采样，$\boldsymbol{\epsilon}$ 从高斯分布采样，$t$ 从 $\{1, \ldots, T\}$ 均匀采样 |
| $\boldsymbol{\epsilon}$ | **实际添加的噪声**（真值） | 在前向过程中随机采样的标准高斯噪声 |
| $\boldsymbol{\epsilon}_\theta(x_t, t)$ | **网络预测的噪声** | 网络的输出，试图估计 $x_t$ 中包含的噪声成分 |
| $\|\cdot\|^2$ | L2 范数平方 | 衡量预测噪声与真实噪声的差距 |

**训练流程伪代码：**

```python
for x_0 in dataloader:                # 1. 从数据集取一批真实图像
    t = randint(1, T)                  # 2. 随机采样时间步 t ∈ {1,...,T}
    ε = torch.randn_like(x_0)         # 3. 采样标准高斯噪声 ε ~ N(0, I)

    # 4. 一步加噪：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * ε

    # 5. 网络预测噪声
    ε_pred = model(x_t, t)

    # 6. 计算损失并反向传播
    loss = F.mse_loss(ε_pred, ε)
    loss.backward()
```

---

### 1.5 Classifier-Free Guidance（CFG）

CFG 是让生成结果更好地遵循文本提示的核心技术。

#### 核心思想

同时训练「有条件生成」和「无条件生成」，推理时将两者**混合放大**：

$$
\boxed{\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(x_t, t, \varnothing) + s \cdot \left[ \boldsymbol{\epsilon}_\theta(x_t, t, c) - \boldsymbol{\epsilon}_\theta(x_t, t, \varnothing) \right]}
$$

等价形式：

$$
\tilde{\boldsymbol{\epsilon}} = (1 - s) \cdot \boldsymbol{\epsilon}_{\text{uncond}} + s \cdot \boldsymbol{\epsilon}_{\text{cond}}
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $\boldsymbol{\epsilon}_\theta(x_t, t, c)$ | **有条件噪声预测** | 给定文本条件 $c$（如 "a cat"）时，网络预测的噪声 |
| $\boldsymbol{\epsilon}_\theta(x_t, t, \varnothing)$ | **无条件噪声预测** | 不给任何文本条件时，网络预测的噪声 |
| $c$ | **文本条件** | 用户输入的 prompt，经 CLIP 编码后的嵌入向量 |
| $\varnothing$ | **空条件** | 空字符串或零向量，代表无条件输入 |
| $s$ | **引导强度**（guidance scale） | 通常取 $7.0 \sim 12.0$；$s=1$ 为标准条件生成 |
| $\tilde{\boldsymbol{\epsilon}}$ | **引导后的噪声预测** | 用于实际去噪采样的最终噪声估计 |

**直觉理解：**

- 「条件预测 $-$ 无条件预测」$= $ 文本条件带来的**方向偏移**
- 乘以 $s > 1$ 就是在这个方向上**走得更远**，更加贴合文本描述
- $s$ 越大 → 越忠于提示词，但多样性和自然度下降

**训练技巧：** 训练时随机以 10%\~20% 的概率将条件 $c$ 替换为 $\varnothing$，使同一个模型同时学会有条件和无条件生成。

---

### 1.6 潜空间扩散（Latent Diffusion / Stable Diffusion）

#### 问题

直接在像素空间做扩散，$512 \times 512 \times 3$ 的图像维度高达 $786{,}432$，计算量巨大。

#### 解决方案

先用预训练的 **VAE** 将图像压缩到低维隐空间，在隐空间做扩散：

$$
\underbrace{512 \times 512 \times 3}_{\text{像素空间}} \xrightarrow{\text{VAE Encoder}} \underbrace{64 \times 64 \times 4}_{\text{隐空间}} \xrightarrow{\text{VAE Decoder}} \underbrace{512 \times 512 \times 3}_{\text{像素空间}}
$$

- 空间分辨率压缩 $8 \times$，计算量降低约 **50 倍**
- VAE 训练目标以**重建质量**为主，KL 散度权重设置极小

#### Stable Diffusion 完整架构

```
          文本 "a cat on the moon"
                    │
                    ▼
         ┌──────────────────┐
         │ CLIP Text Encoder │  → 文本嵌入 (77 × 768)
         └──────────────────┘
                    │
                    ▼ Cross Attention
         ┌──────────────────┐
  z_T ──▶│     UNet 去噪     │──▶ z_0（去噪后的隐码）
 (噪声)  │  （预测噪声 ε_θ）  │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │   VAE Decoder     │  → 生成图像 512 × 512
         └──────────────────┘
```

**三大组件：**

| 组件 | 作用 | 输入/输出 |
|------|------|----------|
| **VAE** | 像素空间 ↔ 隐空间转换 | $512^2 \times 3 \leftrightarrow 64^2 \times 4$ |
| **CLIP Text Encoder** | 文本 → 语义嵌入 | 文本字符串 → $(77, 768)$ 张量 |
| **UNet** | 在隐空间中去噪 | $(z_t, t, \text{text\_emb}) \rightarrow \boldsymbol{\epsilon}_\theta$ |

---

### 1.7 采样器加速

所有采样器可统一表示为：

$$
x_{t-1} = a(t) \cdot \hat{x}_0 + b(t) \cdot \boldsymbol{\epsilon}_\theta + c(t) \cdot \mathbf{z}
$$

| 符号 | 含义 | 说明 |
|------|------|------|
| $a(t), b(t)$ | 确定性系数 | 各采样器不同的计算方式 |
| $c(t)$ | 随机项系数 | $c(t) = 0$ → 确定性采样（ODE）；$c(t) > 0$ → 随机采样（SDE） |
| $\hat{x}_0$ | 从 $\boldsymbol{\epsilon}_\theta$ 反推的 $x_0$ 估计 | $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}$ |

#### 各采样器对比

| 采样器 | 类型 | 步数 | 核心原理 | 适用场景 |
|--------|------|------|---------|---------|
| **DDPM** | 随机 | ~1000 | 原始逐步去噪，含随机项 | 理论基准 |
| **DDIM** | 确定性 | 20-50 | 非马尔可夫过程，可跳步 | 日常生成 |
| **DPM-Solver++** | 确定性 | 15-25 | 高阶 ODE 求解器 | 推荐日常使用 |
| **Euler** | 确定性 | 20-30 | 一阶欧拉法 | 快速预览 |
| **LCM** | 确定性 | 1-4 | 一致性蒸馏 | 实时应用 |

#### DDIM 的关键突破

DDIM 发现：可以定义一族**非马尔可夫过程**，它们拥有相同的边际分布 $q(x_t \mid x_0)$，但允许跨步采样：

$$
\text{DDPM:}\quad x_{1000} \to x_{999} \to \cdots \to x_0 \quad \text{（1000步）}
$$

$$
\text{DDIM:}\quad x_{1000} \to x_{950} \to x_{900} \to \cdots \to x_0 \quad \text{（20步）}
$$

通过参数 $\eta$ 控制随机性：$\eta = 1$ 等价于 DDPM，$\eta = 0$ 为完全确定性采样。

---

### 1.8 高级应用：ControlNet 与 LoRA

#### ControlNet

```
                原始 UNet（冻结）
                ┌─────────────┐
  x_t ─────────▶│  Encoder    │──▶ Decoder ──▶ 输出
                └─────────────┘
                       ↑ Zero Conv（初始权重=0）
                ┌─────────────┐
  条件图 ───────▶│ Encoder 副本 │  ← 可训练
 (边缘/姿态等)   └─────────────┘
```

- 复制 UNet 编码器部分作为**可训练副本**
- 通过 **Zero Convolution**（权重初始化为 0）连接到原始模型
- 训练初期输出为 0，不破坏原模型；逐步学习有意义的条件控制

#### LoRA（Low-Rank Adaptation）

对权重更新做低秩分解：

$$
\Delta W = B \times A, \quad W' = W + \alpha \cdot B \times A
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $W \in \mathbb{R}^{d \times d}$ | 原始权重矩阵 | 如 $768 \times 768$，共 589,824 参数 |
| $A \in \mathbb{R}^{d \times r}$ | 降维矩阵 | $r$ 为秩，通常 $r = 4 \sim 8$ |
| $B \in \mathbb{R}^{r \times d}$ | 升维矩阵 | 与 $A$ 配合完成低秩近似 |
| $r$ | **秩**（rank） | 当 $r = 8$，$d = 768$ 时，参数量仅 $2 \times 768 \times 8 = 12{,}288$（压缩 98%） |
| $\alpha$ | **缩放系数** | 控制 LoRA 对原始权重的影响强度 |

---

## 二、Flow Matching

### 2.1 动机：为什么需要 Flow Matching？

扩散模型效果优秀，但存在固有问题：

| 问题 | 扩散模型的现状 |
|------|--------------|
| 路径形状 | 弯曲路径（受噪声调度 $\beta_t$ 影响），需要更多采样步数 |
| 公式复杂度 | 需要 $\alpha_t, \bar{\alpha}_t, \beta_t, \sigma_t$ 等多组系数 |
| 理论框架 | 绑定高斯噪声，灵活性有限 |

**Flow Matching 的核心主张：用「直线」代替「曲线」。**

```
扩散模型的路径（弯曲）：         Flow Matching 的路径（直线）：

  noise ·                         noise ·
         \                               \
          \                               \
           )                               \
          /                                 \
         /                                   \
  data  ·                         data       ·
```

直线路径意味着：更短的距离、更少的步数、更简单的数学。

---

### 2.2 连续归一化流（CNF）回顾

Flow Matching 建立在连续归一化流的框架上。CNF 通过一个常微分方程（ODE）定义从简单分布到复杂分布的变换：

$$
\frac{dx}{dt} = v_\theta(x, t), \qquad t \in [0, 1]
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $x$ | 样本点 | 在概率空间中流动的「粒子」 |
| $t$ | 连续时间 | 从 $0$（噪声分布）到 $1$（数据分布） |
| $v_\theta(x, t)$ | **速度场**（velocity field） | 神经网络输出，告诉粒子在 $(x, t)$ 处往哪走、走多快 |
| $\theta$ | 网络参数 | 通过训练学习 |
| $x(0) \sim p_0$ | 起始分布 | 简单分布，通常为标准高斯 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| $x(1) \sim p_1$ | 目标分布 | 真实数据分布 |

**直觉**：想象空间中无数粒子，$v_\theta$ 是每个位置的「风场」。粒子随风从 $t=0$ 飘到 $t=1$，就从噪声变成了数据。

**传统 CNF 的问题**：训练需要计算速度场的散度 $\nabla \cdot v_\theta$，计算代价大且数值不稳定。

---

### 2.3 条件 Flow Matching（CFM）

#### 核心突破

**不再计算散度**，而是直接用 MSE 回归目标速度场。

#### 2.3.1 路径定义

给定一对样本 $(x_0, x_1)$，其中 $x_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 为噪声，$x_1$ 为真实数据，定义**线性插值路径**：

$$
\boxed{x_t = (1 - t) \cdot x_0 + t \cdot x_1, \qquad t \in [0, 1]}
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $x_0$ | **噪声样本** | 从标准高斯 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样 |
| $x_1$ | **真实数据样本** | 从训练数据集采样 |
| $t$ | **连续时间** | $t \in [0, 1]$，均匀采样 |
| $(1 - t)$ | 噪声的权重 | $t=0$ 时为 1（纯噪声），$t=1$ 时为 0 |
| $t$ | 数据的权重 | $t=0$ 时为 0，$t=1$ 时为 1（纯数据） |
| $x_t$ | **插值点** | 噪声与数据之间的中间状态 |

**与 Diffusion 对比：**

| | Diffusion | Flow Matching |
|---|-----------|---------------|
| 插值公式 | $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$ | $x_t = (1-t) \cdot x_0 + t \cdot x_1$ |
| 系数 | 非线性（含平方根和累积乘积） | 线性（简单的 $t$ 和 $1-t$） |
| 路径 | 曲线 | 直线 |

#### 2.3.2 目标速度

对路径 $x_t$ 关于 $t$ 求导：

$$
u_t = \frac{dx_t}{dt} = \frac{d}{dt}\left[(1-t) \cdot x_0 + t \cdot x_1\right] = x_1 - x_0
$$

$$
\boxed{u_t(x_t \mid x_0, x_1) = x_1 - x_0}
$$

**极其简洁**：目标速度是一个**常向量**，就是从噪声 $x_0$ 指向数据 $x_1$ 的方向，与时间 $t$ 无关。

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $u_t$ | **目标速度**（ground truth velocity） | 网络的学习目标 |
| $x_1 - x_0$ | 从噪声到数据的位移向量 | 方向 = 数据 $-$ 噪声，大小 = 两者距离 |

#### 2.3.3 训练目标

$$
\boxed{\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\, x_0,\, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]}
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $v_\theta(x_t, t)$ | **网络预测的速度** | 输入插值点 $x_t$ 和时间 $t$，输出速度向量 |
| $x_1 - x_0$ | **目标速度**（真值） | 从噪声到数据的方向 |
| $\|\cdot\|^2$ | L2 范数平方 | 衡量预测速度与目标速度的差距 |
| $\mathbb{E}_{t, x_0, x_1}$ | 期望 | $t \sim \mathcal{U}(0,1)$，$x_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，$x_1 \sim p_{\text{data}}$ |

**训练流程伪代码：**

```python
for x_1 in dataloader:                # 1. 取一批真实数据
    x_0 = torch.randn_like(x_1)      # 2. 采样高斯噪声 x_0 ~ N(0, I)
    t = torch.rand(batch_size)        # 3. 随机时间 t ~ U(0, 1)

    # 4. 线性插值
    x_t = (1 - t) * x_0 + t * x_1

    # 5. 目标速度
    u_t = x_1 - x_0

    # 6. 网络预测速度
    v_pred = model(x_t, t)

    # 7. 计算损失
    loss = F.mse_loss(v_pred, u_t)
    loss.backward()
```

**对比 Diffusion 训练：**

| 步骤 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 预测目标 | 噪声 $\boldsymbol{\epsilon}$ | 速度 $v = x_1 - x_0$ |
| 时间采样 | $t \in \{1, \ldots, 1000\}$ 离散 | $t \in [0, 1]$ 连续 |
| 插值系数 | $\sqrt{\bar{\alpha}_t}$, $\sqrt{1-\bar{\alpha}_t}$ | $(1-t)$, $t$ |
| 额外超参 | 噪声调度 $\beta_t$ | 无 |

---

### 2.4 最优传输 CFM（OT-CFM）

#### 问题：路径交叉

标准 CFM 中 $x_0$ 和 $x_1$ 随机配对，可能导致不同样本的传输路径**交叉**，使速度场在交叉处矛盾：

```
随机配对（路径交叉）：            最优传输配对（路径平行）：

  x_0^a ──────╲╱────── x_1^b       x_0^a ─────────── x_1^a
  x_0^b ──────╱╲────── x_1^a       x_0^b ─────────── x_1^b
          路径交叉！                      路径不交叉！
```

#### 解决方案

在每个 mini-batch 内，用**最优传输**算法找到使总传输距离最短的配对：

$$
\pi^* = \arg\min_{\pi} \sum_{i=1}^{n} \left\| x_0^{(i)} - x_1^{(\pi(i))} \right\|^2
$$

**参数详解：**

| 符号 | 含义 | 说明 |
|------|------|------|
| $\pi$ | **排列函数**（permutation） | 定义哪个噪声样本配对哪个数据样本 |
| $\pi^*$ | **最优排列** | 使总距离最小的配对方案 |
| $x_0^{(i)}$ | 第 $i$ 个噪声样本 | batch 中的噪声 |
| $x_1^{(\pi(i))}$ | 与 $x_0^{(i)}$ 配对的数据样本 | 由排列 $\pi$ 决定 |
| $n$ | batch 大小 | |

实际实现：用 `scipy.optimize.linear_sum_assignment`（匈牙利算法）或 Sinkhorn 算法求解。

**Stable Diffusion 3** 使用的 Rectified Flow 正是采用了 OT 配对策略。

---

### 2.5 Rectified Flow

Rectified Flow 在 OT-CFM 基础上更进一步——通过**迭代 Reflow** 操作逐步拉直路径：

```
第 1 轮：训练模型 v_θ^(1)，学习（可能弯曲的）速度场
         ↓
第 2 轮：用 v_θ^(1) 生成新的 (x_0, x_1) 配对
         重新训练 v_θ^(2) → 路径更直
         ↓
第 3 轮：继续 Reflow → 路径近乎直线
         ↓
最终效果：1-2 步即可生成高质量样本
```

**原理**：每轮 Reflow 都会让模型学到的路径更接近直线，因为新的训练对 $(x_0, x_1)$ 是由上一轮模型沿学到的路径生成的——天然更加对齐。

---

### 2.6 采样过程

采样即**数值求解 ODE**，从纯噪声 $x_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 沿速度场积分到 $t = 1$：

#### Euler 方法（一阶）

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)
$$

| 符号 | 含义 |
|------|------|
| $x_t$ | 当前时刻的样本 |
| $\Delta t$ | 步长，如 $\Delta t = 1/N$（$N$ 为总步数） |
| $v_\theta(x_t, t)$ | 网络预测的速度 |
| $x_{t+\Delta t}$ | 下一时刻的样本 |

#### Heun 方法（二阶，更精确）

$$
\begin{aligned}
k_1 &= v_\theta(x_t,\, t) \\
k_2 &= v_\theta(x_t + \Delta t \cdot k_1,\, t + \Delta t) \\
x_{t + \Delta t} &= x_t + \frac{\Delta t}{2} \cdot (k_1 + k_2)
\end{aligned}
$$

| 符号 | 含义 |
|------|------|
| $k_1$ | 起点处的速度估计 |
| $k_2$ | 终点处的速度估计（用 $k_1$ 预测的终点） |
| $\frac{k_1 + k_2}{2}$ | 两端速度的平均，更精确 |

因为 Flow Matching 的路径更直，**相同步数下精度更高**，或者说**达到相同质量需要更少步数**。

---

## 三、Diffusion vs Flow Matching 全面对比

| 维度 | Diffusion Model | Flow Matching |
|------|----------------|---------------|
| **路径形状** | 弯曲（受噪声调度影响） | 直线（线性插值） |
| **预测目标** | 噪声 $\boldsymbol{\epsilon}$ | 速度 $v$ |
| **时间范围** | $t \in \{1, 2, \ldots, T\}$（离散，$T=1000$） | $t \in [0, 1]$（连续） |
| **核心公式** | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$ | $x_t = (1-t)\, x_0 + t\, x_1$ |
| **训练损失** | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ | $\|v_\theta - (x_1 - x_0)\|^2$ |
| **噪声调度** | 需要精心设计 $\beta_t$ 序列 | 不需要 |
| **额外系数** | $\alpha_t, \bar{\alpha}_t, \beta_t, \sigma_t$ | 无 |
| **采样效率** | 原生需 1000 步，需 DDIM 等加速器 | 天然更少步数 |
| **理论关系** | Flow Matching 的**特例** | **更一般**的框架 |
| **代表模型** | SD 1.5, SDXL | SD 3, Flux |

**数学上的统一**：扩散模型可以被重新表述为一种特殊的 Flow Matching——路径不是直线，而是由噪声调度 $\beta_t$ 决定的曲线。两者可以互相转换。

---

## 四、采样器统一视角

无论是扩散模型还是 Flow Matching，采样本质上都是求解 ODE 或 SDE：

$$
x_{t-1} = a(t) \cdot \hat{x}_0 + b(t) \cdot \boldsymbol{\epsilon}_\theta + c(t) \cdot \mathbf{z}
$$

- $c(t) = 0$ → **确定性采样**（ODE）
- $c(t) > 0$ → **随机采样**（SDE）

**实践推荐：**

| 场景 | 推荐采样器 | 步数 |
|------|-----------|------|
| 日常生成 | DPM++ 2M Karras | 20-30 |
| 快速预览 | Euler | 15-20 |
| 最高质量 | DPM++ 2M SDE Karras | 25-40 |
| 实时应用 | LCM / SDXL Turbo | 1-4 |
| Flow Matching 模型 | Euler / Heun | 20-30 |

---

## 五、推荐学习路径

```
Level 1 ─ 基础
  │
  ├── 理解 GAN / VAE / Diffusion 三大范式的区别
  ├── 掌握前向加噪公式的推导
  └── 理解「预测噪声」的训练目标
  │
Level 2 ─ 核心
  │
  ├── Stable Diffusion 架构（VAE + CLIP + UNet）
  ├── Classifier-Free Guidance 原理
  └── DDIM 采样与加速
  │
Level 3 ─ 进阶
  │
  ├── 潜空间扩散的数学细节
  ├── ControlNet / LoRA / IP-Adapter
  └── DPM-Solver 高阶采样器
  │
Level 4 ─ 前沿
  │
  ├── Flow Matching 理论框架
  ├── 最优传输与 Rectified Flow
  ├── SD3 / Flux 架构（MMDiT）
  └── 视频 / 3D / 音频生成
```

---

> **参考仓库**：[Zhiying-Li-dot/MultiModel](https://github.com/Zhiying-Li-dot/MultiModel)
>
> 仓库中包含更完整的代码实现和多模态大模型笔记，建议配合阅读。
