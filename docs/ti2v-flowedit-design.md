# TI2V FlowEdit 技术方案

## 1. 背景

### 1.1 问题

之前的实验发现：
- **Pure FlowEdit**：没有图像条件，只靠 prompt 引导，效果差
- **Flux.2 + TI2V**：组合方法，Stage 2 无法参考源视频后续帧，背景随时间漂移

### 1.2 目标

结合两者优势：
- 使用 Flux.2 生成高质量的目标第一帧
- 使用 FlowEdit 保持源视频的运动和背景
- 通过 TI2V 模式让目标第一帧引导生成

## 2. 方法概述

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flux.2 + TI2V FlowEdit                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Flux.2 Multi-Reference Edit                           │
│  ─────────────────────────────────────                          │
│  源视频第一帧 + 目标产品图 ──► Flux.2 Dev ──► 目标第一帧        │
│                                                                  │
│  Stage 2: TI2V FlowEdit                                         │
│  ──────────────────────────                                      │
│  源视频 + 目标第一帧 ──► FlowEdit (TI2V) ──► 目标视频           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 核心算法

### 3.1 FlowEdit 回顾

FlowEdit 是一种 training-free 视频编辑方法，核心思想是在 flow matching 框架下，通过源视频和目标 prompt 的差异来引导编辑。

**原始 FlowEdit 算法**：

```python
# 初始化
Zt_edit = X0_src  # 从源视频 latent 开始

for t in timesteps:
    # 1. Forward diffusion on source
    noise = randn_like(X0_src)
    Zt_src = (1 - t) * X0_src + t * noise

    # 2. Compute target latent
    Zt_tar = Zt_edit + Zt_src - X0_src

    # 3. Get flow predictions with CFG
    Vt_src = CFG(Transformer(Zt_src, source_prompt))
    Vt_tar = CFG(Transformer(Zt_tar, target_prompt))

    # 4. Update edit latent
    Zt_edit = Zt_edit + (t_next - t) * (Vt_tar - Vt_src)
```

### 3.2 TI2V 模式

Wan2.2-TI2V-5B 使用 `expand_timesteps` 模式进行图像条件生成：

```python
# 第一帧条件准备
condition = VAE.encode(first_frame)  # [B, 48, 1, H, W]

# 创建 mask (0=用条件，1=用 latent)
mask = ones(1, 1, F, H, W)
mask[:, :, 0] = 0  # 第一帧用条件

# 应用条件
latent_input = mask * latent + (1 - mask) * condition

# 时间步扩展 (第一帧 t=0)
timestep = mask[0, 0, :, ::2, ::2].flatten() * t
```

### 3.3 TI2V FlowEdit 算法

**关键改动**：在 FlowEdit 的每个分支上应用不同的第一帧条件

```python
# 准备条件
source_condition = VAE.encode(source_video_frame1)  # 源视频第一帧
target_condition = VAE.encode(flux_generated_frame)  # Flux.2 生成帧

# FlowEdit 循环
for t in timesteps:
    # 1. Forward diffusion
    Zt_src = (1 - t) * X0_src + t * noise
    Zt_tar = Zt_edit + Zt_src - X0_src

    # 2. Apply TI2V conditioning (关键改动!)
    Zt_src_cond = mask * Zt_src + (1 - mask) * source_condition
    Zt_tar_cond = mask * Zt_tar + (1 - mask) * target_condition

    # 3. Get flow predictions
    Vt_src = CFG(Transformer(Zt_src_cond, source_prompt))
    Vt_tar = CFG(Transformer(Zt_tar_cond, target_prompt))

    # 4. Update
    Zt_edit = Zt_edit + (t_next - t) * (Vt_tar - Vt_src)
```

## 4. 数据流图

```
                    ┌──────────────────┐
                    │   Flux.2 Dev     │
                    │ (32B, bfloat16)  │
                    └────────┬─────────┘
                             │
    ┌────────────┐           │           ┌────────────┐
    │ 源视频第一帧 │───────────┼───────────│ 目标产品图  │
    └─────┬──────┘           │           └──────┬─────┘
          │                  ▼                   │
          │         ┌──────────────┐             │
          │         │ 目标第一帧    │◄────────────┘
          │         │ (Flux生成)   │
          │         └──────┬───────┘
          │                │
          ▼                ▼
    ┌──────────────────────────────────────────────┐
    │              Wan2.2-TI2V-5B                  │
    │         (5B, bfloat16, 48ch latent)          │
    ├──────────────────────────────────────────────┤
    │                                              │
    │  ┌─────────────┐      ┌─────────────┐       │
    │  │ Source Cond │      │ Target Cond │       │
    │  │ (VAE enc)   │      │ (VAE enc)   │       │
    │  └──────┬──────┘      └──────┬──────┘       │
    │         │                    │               │
    │         ▼                    ▼               │
    │  ┌─────────────────────────────────────┐    │
    │  │         FlowEdit Denoising          │    │
    │  │                                     │    │
    │  │  Zt_src ◄── source_cond (frame 0)  │    │
    │  │  Zt_tar ◄── target_cond (frame 0)  │    │
    │  │                                     │    │
    │  │  Zt_edit += Δt * (Vt_tar - Vt_src) │    │
    │  └─────────────────────────────────────┘    │
    │                    │                         │
    │                    ▼                         │
    │            ┌──────────────┐                  │
    │            │  VAE Decode  │                  │
    │            └──────┬───────┘                  │
    └───────────────────┼──────────────────────────┘
                        │
                        ▼
                 ┌────────────┐
                 │  目标视频   │
                 └────────────┘
```

## 5. 实现细节

### 5.1 模型配置

| 组件 | 模型 | 参数量 | 精度 |
|------|------|--------|------|
| Stage 1 | Flux.2 Dev | 32B | bfloat16 |
| Stage 2 | Wan2.2-TI2V-5B | 5B | bfloat16 |
| VAE | Wan2.2 VAE | - | float32 |

### 5.2 内存优化

由于 5B 模型较大，需要：

```python
# 使用 sequential CPU offload
pipe.enable_sequential_cpu_offload(gpu_id=0)

# VAE 单独管理
pipe.vae.to("cpu")  # 编码/解码时临时移到 GPU

# 分辨率限制
height, width = 480, 832  # 或更小
max_frames = 49  # 根据显存调整
```

### 5.3 Timestep 处理

TI2V 模式下，timestep 需要特殊处理：

```python
if expand_timesteps:
    # 第一帧 t=0，其他帧正常
    temp_ts = (first_frame_mask[0, 0, :, ::2, ::2] * t).flatten()
    timestep = temp_ts.unsqueeze(0)
else:
    timestep = t.unsqueeze(0).expand(batch_size)
```

### 5.4 CFG 实现

FlowEdit 需要 4 次 transformer forward：

```python
# 合并输入减少显存峰值
concat_latent = torch.stack([
    Zt_src_uncond, Zt_tar_uncond,  # 无条件
    Zt_src_cond, Zt_tar_cond,      # 有条件
])

# 单次 forward
concat_pred = Transformer(concat_latent, concat_prompts)

# 拆分并计算 CFG
Vt_src = uncond_src + scale * (cond_src - uncond_src)
Vt_tar = uncond_tar + scale * (cond_tar - uncond_tar)
```

## 6. 预期效果

| 特性 | Pure FlowEdit | Flux+TI2V | TI2V FlowEdit |
|------|--------------|-----------|---------------|
| 产品替换 | ❌ 差 | ✅ 好 | ✅ 好 |
| 运动保持 | ✅ 完美 | ✅ 正确 | ✅ 完美 |
| 背景一致 | ✅ 完美 | ❌ 漂移 | ✅ 完美 |
| 第一帧控制 | ❌ 无 | ✅ 有 | ✅ 有 |

## 7. 文件结构

```
pvtt/
├── docs/
│   └── ti2v-flowedit-design.md  # 本文档
└── baseline/compositional-flux-ti2v/
    ├── scripts/
    │   ├── extract_frame.py         # 提取第一帧
    │   ├── flux_edit.py             # Stage 1: Flux.2 编辑
    │   ├── ti2v_generate.py         # 原 TI2V 生成 (对比用)
    │   └── ti2v_flowedit.py         # Stage 2: TI2V FlowEdit (新)
    └── README.md
```

## 8. 测试计划

1. **基础验证**：bracelet → pearl necklace
2. **对比实验**：
   - Pure FlowEdit (无图像条件)
   - Flux + TI2V (组合方法)
   - TI2V FlowEdit (本方案)
3. **评估指标**：
   - 产品替换准确度
   - 运动保持度 (光流一致性)
   - 背景一致性 (SSIM)

## 9. 参考

- [FlowEdit](https://github.com/fallenshock/FlowEdit) - 原始 FlowEdit 实现
- [Awesome-Training-Free-WAN2.1-Editing](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing) - WAN 适配
- [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) - TI2V 模型
