# Flow Matching Inversion + TI2V 技术方案

## 背景

### 问题回顾

之前的 TI2V + FlowEdit 实验失败，原因是 FlowEdit 的 **Inversion-Free** 设计导致源视频内容被显式保留在 latent 中：

```python
# FlowEdit Inversion-Free
Zt_src = (1-t) * X0_src + t * noise  # 显式包含源视频
Zt_tar = Zt_edit + (Zt_src - X0_src)  # 后续帧仍含源视频结构
```

当用目标首帧替换 `Zt_tar` 的首帧时，后续帧的 latent 仍然包含源视频内容，导致：
- 首帧正确（目标首帧）
- 后续帧退化为无图像条件状态

### 解决思路

使用传统的 **Flow Matching Inversion** 替代 Inversion-Free：
1. 先将源视频 **反演（Inversion）** 到纯噪声空间
2. 从反演得到的噪声出发，用 TI2V 模型 **去噪生成** 目标视频

关键区别：
- **Inversion-Free**: latent 显式包含源视频内容
- **Inversion**: 噪声只隐式编码结构信息（运动、布局），不含具体内容

```
┌─────────────────────────────────────────────────────────────────┐
│                    传统 Inversion 流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   源视频 X0_src    Inversion     噪声 ZT      TI2V 去噪    目标视频   │
│   ┌─────────┐    ─────────>   ┌─────────┐   ─────────>  ┌─────────┐│
│   │ 🖼 手链 │      t: 0→1     │ 📊 噪声 │     t: 1→0    │ 🖼 项链 ││
│   │ 🖼 手链 │  (source_prompt)│ 📊 噪声 │  + 目标首帧   │ 🖼 项链 ││
│   │ 🖼 手链 │                 │ 📊 噪声 │  + target_prompt│ 🖼 项链 ││
│   └─────────┘                 └─────────┘              └─────────┘│
│                                                                 │
│   噪声 ZT 只编码结构（运动模式），不含源视频具体内容              │
│   TI2V 去噪时根据目标首帧生成一致的后续帧                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Wan2.2 TI2V 代码分析

### 核心组件

| 组件 | 类/文件 | 说明 |
|------|---------|------|
| VAE | `Wan2_2_VAE` | 视频编解码，stride=(4,16,16) |
| Text Encoder | `T5EncoderModel` | 文本编码 |
| DiT Model | `WanModel` | 视频生成 Transformer |
| Scheduler | `FlowDPMSolverMultistepScheduler` | Flow Matching ODE 求解器 |

### TI2V (i2v) 推理流程

```python
# 1. 编码首帧
z = self.vae.encode([img])  # img: [3, 1, H, W] → z: [16, 1, H/16, W/16]

# 2. 创建 mask（首帧=0，后续帧=1）
mask1, mask2 = masks_like([noise], zero=True)
# mask2[:, 0] = 0, mask2[:, 1:] = 1

# 3. 初始化 latent
latent = (1. - mask2[0]) * z[0] + mask2[0] * noise
# 首帧是编码后的图像，后续帧是噪声

# 4. 去噪循环
for t in timesteps:  # t: 1 → 0 (sigma_max → sigma_min)
    # CFG
    v_cond = model(latent, t, context)
    v_uncond = model(latent, t, context_null)
    v = v_uncond + scale * (v_cond - v_uncond)

    # ODE step
    latent = scheduler.step(v, t, latent)

    # 保持首帧固定
    latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
```

### Flow Matching ODE

Wan2.2 使用 Flow Matching，ODE 形式：

$$\frac{dx}{dt} = v_\theta(x_t, t)$$

其中：
- **去噪（Denoising）**: $t: 1 \to 0$，从噪声到干净数据
- **反演（Inversion）**: $t: 0 \to 1$，从干净数据到噪声

Scheduler 中的关键变量：
- `sigma`: 噪声水平，范围 [0, 1]
- `timesteps = sigmas * num_train_timesteps`
- 去噪时 `sigmas` 从大到小
- 反演时 `sigmas` 从小到大（翻转）

---

## 技术方案

### 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 编码源视频                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ z0_src = vae.encode(source_video)   # [16, F, H/16, W/16]           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Inversion（反演）                                            │
├─────────────────────────────────────────────────────────────────────┤
│ # 翻转 timesteps: t: 0 → 1                                          │
│ sigmas_inv = flip(sigmas)  # [sigma_min, ..., sigma_max]            │
│                                                                     │
│ zT = z0_src                                                         │
│ for t in timesteps_inv:                                             │
│     v = model(zT, t, source_context)  # 使用源视频 prompt            │
│     zT = scheduler_inv.step(v, t, zT)  # ODE step (向噪声方向)       │
│                                                                     │
│ # 结果：zT 是反演得到的噪声                                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: TI2V 去噪（使用目标首帧）                                      │
├─────────────────────────────────────────────────────────────────────┤
│ # 编码目标首帧                                                        │
│ z_target_first = vae.encode(target_first_frame)  # [16, 1, H/16, W/16]│
│                                                                     │
│ # 创建 mask                                                          │
│ mask2 = ones_like(zT)                                               │
│ mask2[:, 0] = 0  # 首帧 mask = 0                                    │
│                                                                     │
│ # 用目标首帧替换反演噪声的首帧                                         │
│ latent = (1 - mask2) * z_target_first + mask2 * zT                  │
│                                                                     │
│ # 正常 TI2V 去噪                                                      │
│ for t in timesteps:  # t: 1 → 0                                     │
│     v_cond = model(latent, t, target_context)                       │
│     v_uncond = model(latent, t, context_null)                       │
│     v = v_uncond + scale * (v_cond - v_uncond)                      │
│     latent = scheduler.step(v, t, latent)                           │
│     latent = (1 - mask2) * z_target_first + mask2 * latent          │
│                                                                     │
│ # 解码                                                               │
│ target_video = vae.decode(latent)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Inversion 实现要点

#### 1. Timesteps 翻转

```python
# 正常去噪: sigmas 从大到小
sigmas = get_sampling_sigmas(sampling_steps, shift)  # [sigma_max, ..., sigma_min]

# Inversion: sigmas 从小到大（翻转）
sigmas_inv = np.flip(sigmas)  # [sigma_min, ..., sigma_max]
scheduler.set_timesteps(sigmas=sigmas_inv)
```

#### 2. ODE Step 方向

Flow Matching 的 Euler step：
```python
# 去噪 (t: 大→小)
dt = sigma_next - sigma  # dt < 0
x_next = x + v * dt

# Inversion (t: 小→大)
dt = sigma_next - sigma  # dt > 0
x_next = x + v * dt
```

Scheduler 自动处理方向，只需翻转 timesteps。

#### 3. 无 CFG

Inversion 阶段**不使用 CFG**，只用 conditional prediction：
```python
# Inversion 时
v = model(zT, t, source_context)  # 只用 source prompt

# 去噪时正常使用 CFG
v = v_uncond + scale * (v_cond - v_uncond)
```

### 代码结构

```
baseline/compositional-flux-ti2v/scripts/
├── ti2v_flowedit.py          # 现有 FlowEdit 实现
└── ti2v_inversion.py         # 新增：Inversion + TI2V 实现
```

新脚本 `ti2v_inversion.py` 核心函数：

```python
def inversion(
    model, vae, text_encoder, scheduler,
    source_video: torch.Tensor,      # [3, F, H, W]
    source_prompt: str,
    device: torch.device,
    sampling_steps: int = 50,
    shift: float = 5.0,
) -> torch.Tensor:
    """
    将源视频反演到噪声空间

    Returns:
        zT: 反演得到的噪声 [16, F', H', W']
    """
    # 1. 编码源视频
    z0 = vae.encode(source_video)

    # 2. 准备反演 timesteps
    sigmas = get_sampling_sigmas(sampling_steps, shift)
    sigmas_inv = np.flip(sigmas)
    scheduler.set_timesteps(sigmas=sigmas_inv)

    # 3. 编码 source prompt
    context = text_encoder([source_prompt], device)

    # 4. Inversion 循环
    zT = z0
    for t in scheduler.timesteps:
        v = model(zT, t, context)
        zT = scheduler.step(v, t, zT)

    return zT


def ti2v_with_inversion(
    model, vae, text_encoder, scheduler,
    source_video: torch.Tensor,      # [3, F, H, W]
    source_prompt: str,
    target_first_frame: torch.Tensor, # [3, 1, H, W]
    target_prompt: str,
    device: torch.device,
    sampling_steps: int = 50,
    shift: float = 5.0,
    guide_scale: float = 5.0,
) -> torch.Tensor:
    """
    Inversion + TI2V 生成目标视频
    """
    # 1. Inversion
    zT = inversion(model, vae, text_encoder, scheduler,
                   source_video, source_prompt, device,
                   sampling_steps, shift)

    # 2. 编码目标首帧
    z_target_first = vae.encode(target_first_frame)

    # 3. 创建 mask
    mask2 = torch.ones_like(zT)
    mask2[:, 0] = 0

    # 4. 用目标首帧替换
    latent = (1 - mask2) * z_target_first + mask2 * zT

    # 5. TI2V 去噪
    sigmas = get_sampling_sigmas(sampling_steps, shift)
    scheduler.set_timesteps(sigmas=sigmas)
    context_target = text_encoder([target_prompt], device)
    context_null = text_encoder([""], device)

    for t in scheduler.timesteps:
        v_cond = model(latent, t, context_target)
        v_uncond = model(latent, t, context_null)
        v = v_uncond + guide_scale * (v_cond - v_uncond)
        latent = scheduler.step(v, t, latent)
        latent = (1 - mask2) * z_target_first + mask2 * latent

    # 6. 解码
    target_video = vae.decode(latent)
    return target_video
```

---

## 与 FlowEdit 的对比

| | FlowEdit (Inversion-Free) | Inversion + TI2V |
|---|---|---|
| **源视频处理** | 前向扩散 `Zt = (1-t)*X0 + t*noise` | 反演到噪声空间 |
| **latent 内容** | 显式包含源视频 | 只编码结构信息 |
| **目标首帧条件** | ❌ 后续帧退化 | ✅ 应可正确传播 |
| **计算成本** | 低（无反演步骤） | 高（额外反演步骤） |
| **编辑方式** | velocity 差异驱动 | TI2V 首帧条件生成 |

---

## 预期效果

1. **首帧**：与目标首帧一致（项链）
2. **后续帧**：保持首帧内容一致性（都是项链）
3. **运动模式**：继承源视频的运动轨迹和节奏
4. **结构保留**：手部位置、光照、相机运动与源视频相似

---

## 风险与备选方案

### 可能的问题

1. **Inversion 精度不足**
   - Flow Matching Euler inversion 可能有误差累积
   - 备选：使用更高阶的 RF-Solver（二阶 Taylor 展开）

2. **运动模式丢失**
   - 反演噪声可能没有保留足够的运动信息
   - 备选：结合 RF-Solver 的 attention 特征共享

3. **首帧与后续帧不一致**
   - 模型可能无法从目标首帧正确推断后续帧
   - 备选：增加 inversion 步数、调整 shift 参数

### 实验计划

1. **基础实验**：简单 Euler Inversion + TI2V
2. **参数调优**：sampling_steps, shift, guide_scale
3. **对比实验**：与 Flux.2 + TI2V 两阶段方法对比
4. **进阶方案**：如果简单方案效果不佳，考虑 RF-Solver

---

## 参考资料

1. [RF-Solver](https://arxiv.org/abs/2411.04746) - Rectified Flow Inversion
2. [Wan2.2 代码](https://github.com/Wan-AI/Wan2.2) - 官方实现
3. [FlowEdit](https://arxiv.org/abs/2412.08629) - Inversion-Free 编辑方法
