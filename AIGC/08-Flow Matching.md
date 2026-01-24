# 第八章：Flow Matching

Flow Matching 是一种新兴的生成模型训练方法，被 Stable Diffusion 3 等最新模型采用，相比传统扩散模型更简洁高效。

## 8.1 背景：什么是Flow

### 从概率分布变换说起

```
Flow的核心思想：学习一个可逆变换，将简单分布变换为复杂分布

简单分布 (高斯噪声)              复杂分布 (真实数据)
    z ~ N(0, I)      ────────►      x ~ p_data
         │              f_θ              │
         │           (神经网络)           │
    ┌────┴────┐                    ┌────┴────┐
    │ ● ● ●   │                    │  🐱 🐶   │
    │  ● ● ●  │   ══════════►     │ 图像分布  │
    │   ● ●   │                    │         │
    └─────────┘                    └─────────┘
      噪声分布                        数据分布
```

### 连续归一化流 (CNF)

```
CNF用ODE (常微分方程) 描述变换过程：

dx/dt = v_θ(x, t)    t ∈ [0, 1]

其中 v_θ 是神经网络参数化的速度场

时间演化：
t=0: 噪声分布 p_0 = N(0, I)
t=1: 数据分布 p_1 ≈ p_data

┌─────────────────────────────────────────────────────┐
│                                                     │
│   t=0        t=0.25       t=0.5       t=0.75   t=1 │
│    ●           ●           ◐           ◑        ■   │
│   噪声  ───►  ───►  ───►  ───►  ───►  数据    │
│                                                     │
│   粒子沿着速度场 v_θ(x,t) 移动                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 8.2 传统CNF的问题

### 训练困难

```
传统方法需要计算对数似然：

log p_1(x) = log p_0(z) - ∫₀¹ ∇·v_θ(x_t, t) dt

问题：
1. 需要计算散度 ∇·v_θ，计算量大
2. 需要求解ODE来计算积分
3. 训练不稳定，收敛慢

这就是为什么早期Flow模型不如GAN/VAE流行
```

---

## 8.3 Flow Matching：简单而强大

### 核心思想

```
Flow Matching的突破：

不直接学习似然，而是直接回归速度场！

给定：
- 目标速度场 u_t(x)（我们知道的理想速度场）
- 神经网络 v_θ(x, t)

训练目标：
L_FM = E_{t, x_t} [||v_θ(x_t, t) - u_t(x_t)||²]

就是简单的MSE回归！类似扩散模型预测噪声
```

### 为什么这样做是对的？

```
定理：如果 v_θ = u_t（速度场匹配），
     那么 v_θ 生成的分布流 = 目标分布流

直觉：
- 速度场决定了粒子如何移动
- 匹配速度场 = 匹配整个变换过程
- 无需计算似然或散度
```

---

## 8.4 条件Flow Matching (CFM)

### 问题：u_t 从哪来？

```
我们不知道从噪声到数据的"真实"速度场 u_t

解决方案：条件Flow Matching

思路：
1. 对于每个数据点 x_1，定义一条从噪声 x_0 到 x_1 的路径
2. 这条路径的速度场是已知的
3. 对所有数据点的速度场取期望，得到边际速度场
```

### 最简单的路径：线性插值

```
给定：
- x_0 ~ N(0, I)  (噪声)
- x_1 ~ p_data   (数据)

定义条件路径（线性插值）：
x_t = (1-t)·x_0 + t·x_1

对应的条件速度场：
u_t(x | x_0, x_1) = x_1 - x_0

即：从 x_0 直接指向 x_1 的向量

┌─────────────────────────────────────────┐
│                                         │
│     x_0 ●─────────────────────► ● x_1  │
│          ╲                     ╱        │
│           ╲    x_t = 插值点   ╱         │
│            ╲       ●        ╱          │
│             ╲      │       ╱            │
│              ╲     │      ╱             │
│               速度 = x_1 - x_0          │
│                                         │
└─────────────────────────────────────────┘
```

### CFM训练目标

```python
# Conditional Flow Matching Loss
def cfm_loss(model, x_1):
    """
    x_1: 真实数据样本
    """
    # 1. 采样噪声
    x_0 = torch.randn_like(x_1)

    # 2. 采样时间
    t = torch.rand(x_1.shape[0], 1, 1, 1)

    # 3. 计算插值点
    x_t = (1 - t) * x_0 + t * x_1

    # 4. 目标速度（条件速度场）
    u_t = x_1 - x_0

    # 5. 预测速度
    v_t = model(x_t, t)

    # 6. MSE损失
    loss = F.mse_loss(v_t, u_t)

    return loss
```

---

## 8.5 与扩散模型的对比

### 路径对比

```
扩散模型 (DDPM):
─────────────────────────────────────────────────────
x_0 (数据) ──加噪──► x_t ──加噪──► x_T (噪声)

前向：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
反向：预测噪声 ε，逐步去噪

路径是曲线的（受噪声调度影响）


Flow Matching:
─────────────────────────────────────────────────────
x_0 (噪声) ──直线──► x_t ──直线──► x_1 (数据)

路径：x_t = (1-t) · x_0 + t · x_1
预测：速度 v = x_1 - x_0

路径是直线的（最短路径）
```

### 公式对比

```
┌──────────────────┬─────────────────────┬────────────────────┐
│                  │      Diffusion      │   Flow Matching    │
├──────────────────┼─────────────────────┼────────────────────┤
│ 时间方向         │ 1→0 (数据→噪声)     │ 0→1 (噪声→数据)    │
├──────────────────┼─────────────────────┼────────────────────┤
│ 中间状态         │ x_t = √ᾱ·x₀+√(1-ᾱ)·ε│ x_t = (1-t)x₀+t·x₁ │
├──────────────────┼─────────────────────┼────────────────────┤
│ 预测目标         │ 噪声 ε              │ 速度 v = x₁ - x₀   │
├──────────────────┼─────────────────────┼────────────────────┤
│ 损失函数         │ ||ε - ε_θ||²        │ ||v - v_θ||²       │
├──────────────────┼─────────────────────┼────────────────────┤
│ 采样方式         │ 离散去噪步骤        │ ODE求解器          │
└──────────────────┴─────────────────────┴────────────────────┘
```

### 优势对比

```
Flow Matching 的优势：

1. 更直的路径
   - 扩散：曲线路径，需要更多步数
   - FM：直线路径，理论上更高效

2. 更简单的公式
   - 扩散：需要噪声调度 β_t, ᾱ_t
   - FM：只有线性插值

3. 更灵活
   - 可以选择不同的路径（不限于直线）
   - 可以结合最优传输

4. 统一框架
   - 扩散模型是FM的特例
   - 更容易理解和扩展
```

---

## 8.6 最优传输 Flow Matching (OT-CFM)

### 问题：随机配对的低效

```
标准CFM的问题：

x_0 和 x_1 是随机配对的
可能导致路径交叉，效率低

    x_0¹ ●─────────╲─────────● x_1²
                    ╳  (路径交叉!)
    x_0² ●─────────╱─────────● x_1¹

最优传输配对：找到最短总距离的配对

    x_0¹ ●───────────────────● x_1¹

    x_0² ●───────────────────● x_1²

    (无交叉，路径更短)
```

### OT-CFM实现

```python
import ot  # Python Optimal Transport库

def ot_cfm_loss(model, x_1, eps=1e-5):
    """
    使用最优传输配对的CFM
    """
    batch_size = x_1.shape[0]

    # 1. 采样噪声
    x_0 = torch.randn_like(x_1)

    # 2. 计算最优传输矩阵
    # 代价矩阵：欧氏距离
    M = torch.cdist(x_0.flatten(1), x_1.flatten(1)) ** 2
    M = M / M.max()  # 归一化

    # 均匀分布权重
    a = torch.ones(batch_size) / batch_size
    b = torch.ones(batch_size) / batch_size

    # 求解OT（使用Sinkhorn算法）
    pi = ot.sinkhorn(a.numpy(), b.numpy(), M.numpy(), reg=eps)
    pi = torch.from_numpy(pi)

    # 3. 按OT配对重排x_1
    # 采样配对索引
    indices = torch.multinomial(pi, 1).squeeze()
    x_1_matched = x_1[indices]

    # 4. 正常的CFM
    t = torch.rand(batch_size, 1, 1, 1)
    x_t = (1 - t) * x_0 + t * x_1_matched
    u_t = x_1_matched - x_0

    v_t = model(x_t, t)
    loss = F.mse_loss(v_t, u_t)

    return loss
```

### Mini-batch OT

```
实际中使用 Mini-batch OT：

1. 在每个batch内计算OT
2. 计算量可接受：O(n²)对于batch size n
3. 近似全局OT的效果

这是 Stable Diffusion 3 使用的方法
```

---

## 8.7 完整实现

### Flow Matching训练

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingTrainer:
    def __init__(self, model, sigma_min=0.001):
        """
        model: 预测速度的神经网络 v_θ(x, t)
        sigma_min: 最小噪声（用于稳定性）
        """
        self.model = model
        self.sigma_min = sigma_min

    def get_train_tuple(self, x_1):
        """
        获取训练所需的 (x_t, t, target)

        x_1: 真实数据 (batch_size, ...)
        """
        batch_size = x_1.shape[0]

        # 采样噪声 x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # 采样时间 t ~ U(0, 1)
        t = torch.rand(batch_size, device=x_1.device)

        # 扩展t的维度以便广播
        t_expanded = t.view(-1, *([1] * (x_1.dim() - 1)))

        # 计算 x_t（线性插值 + 小噪声保持稳定）
        # x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # 目标速度
        target = x_1 - x_0

        return x_t, t, target

    def training_step(self, x_1):
        """单步训练"""
        x_t, t, target = self.get_train_tuple(x_1)

        # 预测速度
        v_pred = self.model(x_t, t)

        # MSE损失
        loss = F.mse_loss(v_pred, target)

        return loss


class FlowMatchingSampler:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample_euler(self, shape, num_steps=50, device='cuda'):
        """
        欧拉法采样
        """
        # 从噪声开始
        x = torch.randn(shape, device=device)

        # 时间步
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(shape[0], device=device) * (i / num_steps)

            # 预测速度
            v = self.model(x, t)

            # 欧拉更新: x_{t+dt} = x_t + v * dt
            x = x + v * dt

        return x

    @torch.no_grad()
    def sample_heun(self, shape, num_steps=50, device='cuda'):
        """
        Heun方法（二阶）采样，更准确
        """
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            t_next = (i + 1) / num_steps

            t_tensor = torch.ones(shape[0], device=device) * t
            t_next_tensor = torch.ones(shape[0], device=device) * t_next

            # 预测当前速度
            v = self.model(x, t_tensor)

            # 欧拉预测
            x_euler = x + v * dt

            # Heun校正（使用预测点的速度）
            if i < num_steps - 1:
                v_next = self.model(x_euler, t_next_tensor)
                x = x + (v + v_next) / 2 * dt
            else:
                x = x_euler

        return x
```

### 简单的速度网络

```python
class VelocityNetwork(nn.Module):
    """
    简单的MLP速度网络（用于演示）
    实际应用中使用UNet或DiT
    """
    def __init__(self, dim=2, hidden_dim=256, time_dim=64):
        super().__init__()

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 主网络
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t)

        # 拼接输入
        x_input = torch.cat([x, t_emb], dim=-1)

        # 预测速度
        v = self.net(x_input)

        return v


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

---

## 8.8 Stable Diffusion 3 中的应用

### SD3使用Rectified Flow

```
Stable Diffusion 3 的关键改进：

1. Rectified Flow (整流流)
   - 基于Flow Matching
   - 使用直线路径
   - OT配对提高效率

2. MMDiT架构
   - 多模态DiT
   - 图像和文本token联合处理

3. 训练公式
   x_t = (1-t) · ε + t · x
   v = x - ε  (速度 = 数据 - 噪声)

   损失: ||v_θ(x_t, t) - v||²
```

### SD3风格的Flow Matching

```python
class RectifiedFlowTrainer:
    """
    Stable Diffusion 3 风格的Rectified Flow
    """
    def __init__(self, model):
        self.model = model

    def training_step(self, x):
        """
        x: 真实图像 (在潜空间)
        """
        batch_size = x.shape[0]

        # 采样噪声
        noise = torch.randn_like(x)

        # 采样时间 (使用logit-normal分布，SD3的选择)
        # 这会在t≈0.5附近采样更多
        u = torch.randn(batch_size, device=x.device)
        t = torch.sigmoid(u)
        t = t.view(-1, 1, 1, 1)

        # 插值
        x_t = (1 - t) * noise + t * x

        # 目标速度
        v_target = x - noise

        # 预测
        v_pred = self.model(x_t, t.squeeze())

        # 损失（可选：加权）
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(self, shape, num_steps=28, cfg_scale=4.5, **kwargs):
        """
        采样（带CFG）
        """
        x = torch.randn(shape)

        # 时间步（SD3使用特定调度）
        timesteps = torch.linspace(0, 1, num_steps + 1)

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # CFG
            v_uncond = self.model(x, t, condition=None)
            v_cond = self.model(x, t, **kwargs)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            # 欧拉步
            x = x + v * dt

        return x
```

---

## 8.9 Rectified Flow：路径拉直

### Reflow操作

```
Rectified Flow的核心思想：迭代拉直路径

问题：即使用直线路径训练，学到的流可能不是完全直的

Reflow过程：
1. 用当前模型生成 (x_0, x_1) 配对
2. 用这些配对重新训练
3. 重复，路径越来越直

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  第1轮:     x_0 ●~~~~~~~~~~~~~~~~~~~~~~● x_1           │
│                    (学到的路径可能弯曲)                  │
│                                                         │
│  第2轮:     x_0 ●~~~~~~~~~~~~~~~~~~~~● x_1             │
│                    (更直一些)                           │
│                                                         │
│  第3轮:     x_0 ●──────────────────● x_1               │
│                    (接近直线)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘

好处：路径越直，需要的采样步数越少
```

### Reflow实现

```python
class ReflowTrainer:
    """
    Rectified Flow的Reflow训练
    """
    def __init__(self, model):
        self.model = model

    def generate_pairs(self, num_samples, shape):
        """
        用当前模型生成 (x_0, x_1) 配对
        """
        # 采样噪声
        x_0 = torch.randn(num_samples, *shape)

        # 用模型生成x_1
        x_1 = self.sample(x_0, num_steps=100)

        return x_0, x_1

    def reflow_step(self, x_0, x_1):
        """
        用生成的配对训练，拉直路径
        """
        batch_size = x_0.shape[0]

        # 采样时间
        t = torch.rand(batch_size, 1, 1, 1)

        # 直线插值
        x_t = (1 - t) * x_0 + t * x_1

        # 目标：直线速度
        v_target = x_1 - x_0

        # 预测
        v_pred = self.model(x_t, t)

        # 损失
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(self, x_0, num_steps=50):
        """从x_0采样到x_1"""
        x = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(x.shape[0], 1, 1, 1) * (i / num_steps)
            v = self.model(x, t)
            x = x + v * dt

        return x
```

---

## 8.10 总结

### Flow Matching vs Diffusion

```
┌────────────────────────────────────────────────────────────┐
│                    核心差异总结                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Diffusion:                                                │
│  - 基于SDE/随机过程                                        │
│  - 预测噪声 ε                                              │
│  - 需要噪声调度                                            │
│  - 路径由方差调度决定（曲线）                              │
│                                                            │
│  Flow Matching:                                            │
│  - 基于ODE/确定性流                                        │
│  - 预测速度 v                                              │
│  - 无需复杂调度                                            │
│  - 路径可以是直线（更高效）                                │
│                                                            │
│  联系:                                                     │
│  - 数学上可以相互转换                                      │
│  - Diffusion是FM的一种特例                                 │
│  - 两者用同样的网络架构（UNet/DiT）                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 关键公式速查

```
1. 条件路径:     x_t = (1-t)·x_0 + t·x_1

2. 条件速度:     u_t = x_1 - x_0

3. CFM损失:      L = E[||v_θ(x_t, t) - u_t||²]

4. ODE采样:      dx/dt = v_θ(x, t),  从t=0积分到t=1

5. 欧拉采样:     x_{t+dt} = x_t + v_θ(x_t, t)·dt
```

### 学习资源

```
论文:
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Rectified Flow" (Liu et al., 2022)
- "Scaling Rectified Flow Transformers" (SD3, 2024)

代码:
- torchdyn: PyTorch动力系统库
- flow_matching: Facebook官方实现
- diffusers: HuggingFace库支持Flow Matching
```

### 适用场景

| 场景 | 推荐方法 |
|------|----------|
| 需要最高质量 | Diffusion (更成熟) |
| 需要快速采样 | Flow Matching + Reflow |
| 新项目开发 | Flow Matching (更简洁) |
| 使用SD3 | Rectified Flow |
