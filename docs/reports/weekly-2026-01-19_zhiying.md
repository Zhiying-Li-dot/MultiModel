# Video Editing 周报

> **日期**: 2025年1月19日
> **项目**: PVTT - Video Editing Pipeline

---

## 本周工作概述

本周完成了基于 **FLUX.2 Dev + Wan2.2 TI2V** 的两阶段视频编辑Pipeline开发，实现了从图像编辑到视频生成的完整流程。

---

## 两阶段视频编辑Pipeline演示

### Pipeline整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Video Editing Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [原始视频]  ──┬──► [Stage 1] 首帧提取 ──► first_frame.png            │
│                │                                                        │
│                └──► [Stage 2] 相机运动分析 ──► motion_description       │
│                                                                         │
│   [目标参考图] ──┬                                                      │
│                 │                                                       │
│                 ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ [Stage 3] FLUX.2 Dev 多参考图像编辑                              │  │
│   │  • 输入: 原始首帧 + 目标物体参考图                                │  │
│   │  • 输出: edited_frame.png                                       │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                 │                                                       │
│                 ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ [Stage 4] Wan2.2 TI2V-5B 视频生成                                │  │
│   │  • 输入: 编辑后首帧 + 运动描述                                    │  │
│   │  • 输出: output_video.mp4                                       │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: FLUX.2 Dev 多参考图像编辑

### 技术方案

采用 **FLUX.2 Dev** 模型进行多参考图像编辑，支持输入多张参考图实现物体替换。

### 核心代码实现

```python
def flux2_edit(src_img, tgt_img, src_obj, tgt_obj, out_path, model_path):
    """FLUX.2 多参考图像编辑"""
    from diffusers import Flux2Pipeline

    # 加载FLUX.2 Pipeline
    pipe = Flux2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    # 多参考图像Prompt
    prompt = f"""Replace the {src_obj} in image 1 with the {tgt_obj} from image 2.
               Keep the exact same pose, lighting, and background as image 1."""

    # 生成编辑后的图像
    result = pipe(
        image=[src_img, tgt_img],  # 多图输入
        prompt=prompt,
        guidance_scale=4.5,
        num_inference_steps=100,
    ).images[0]

    return result
```

### 效果展示

| 输入 | 输出 |
|------|------|
| **原始首帧** (first_frame.png) | **编辑后图像** (edited_frame.png) |
| 分辨率: 1280 × 1024 | 分辨率: 1280 × 1024 |
| 包含原始物体 (bracelet) | 替换为目标物体 (necklace) |

**关键参数配置：**

| 参数 | 值 |
|------|-----|
| `guidance_scale` | 4.5 |
| `num_inference_steps` | 100 |
| `torch_dtype` | bfloat16 |

---

## Stage 2: Wan2.2 TI2V 视频生成

### 技术方案

采用 **Wan2.2 TI2V-5B** 模型，从编辑后的首帧生成完整视频，保持与原视频一致的相机运动。

### 核心代码实现

```python
def ti2v_generate(first_frame, motion_desc, tgt_obj, out_path, model_path, num_frames=49):
    """Wan2.2 TI2V视频生成"""

    # 加载Wan2.2模型组件
    config = WAN_CONFIGS["ti2v-5B"]
    text_encoder = T5EncoderModel(...)
    vae = Wan2_2_VAE(...)
    model = WanModel.from_pretrained(model_path)

    # 编码首帧
    z_first = vae.encode([first_frame_tensor])

    # 构建生成Prompt，包含运动描述
    prompt = f"Professional product video of {tgt_obj}. Camera {motion_desc}.
               Luxury, cinematic, soft lighting."

    # 文本编码
    context = text_encoder([prompt], device)

    # 去噪生成
    for i, t in enumerate(timesteps):
        # CFG引导
        v_pred = model(z_t, t=timestep, context=context)
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

        # 更新latent
        z_t = scheduler.step(v_guided, t, z_t)

    # 解码生成视频
    output_video = vae.decode([z_t])
    save_video(output_video, out_path, fps=24)
```

### 效果展示

| 参数 | 配置值 |
|------|--------|
| 输出分辨率 | 1056 × 832 |
| 帧数 | 49 frames |
| 帧率 | 24 fps |
| 视频时长 | ~2秒 |
| 编码格式 | H.264 / AVC |

**关键参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `guidance_scale` | 5.0 | CFG强度 |
| `num_inference_steps` | 30 | 去噪步数 |
| `shift` | 5.0 | 时间步偏移 |

---

## 相机运动分析模块

Pipeline中集成了相机运动分析，使用光流法自动检测原视频的运动模式：

```python
def analyze_camera_motion(video_path, max_frames=100):
    """基于光流的相机运动分析"""
    # 使用Farneback光流算法
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, ...)

    # 分析运动分量
    pan  = np.mean(flow[:,:,0])           # 水平平移
    tilt = np.mean(flow[:,:,1])           # 垂直平移
    zoom = np.mean((flow*radial)/dist)    # 缩放
    rot  = np.mean((flow*tangent)/dist²)  # 旋转

    # 生成运动描述
    # 例如: "slow pan to the right, slow zoom in"
```

---

## 输出文件说明

```
pipeline_output/new_video_edit/
├── first_frame.png      # 原始视频首帧 (372 KB, 1280×1024)
├── edited_frame.png     # FLUX.2编辑后 (1.2 MB, 1280×1024)
└── output_video.mp4     # 最终视频 (88 KB, 1056×832, 24fps)
```

---

## 实验迭代历程

本周共完成多轮实验迭代：

| 版本 | 主要改进 |
|------|----------|
| video_edit_v1 | 初版Pipeline搭建 |
| video_edit_v2~v4 | Pipeline集成优化 |
| flux_edit/inpaint | FLUX编辑方法对比 |
| **video_edit_pipeline.py** | **最终整合版本** |

---

## 下周计划

1. 定量评估编辑效果的时序一致性 (CLIP-Score, FVD等)
2. 优化生成参数组合 (guidance_scale的消融实验)
3. 扩展测试更多编辑场景 (bear, blackswan, boat等)
4. 显存优化，支持更高分辨率生成

---

## 代码位置

- **主Pipeline代码**: `/data/lizhiying/Video-Editing/pvtt-main/baseline/compositional-flux-ti2v/scripts/video_edit_pipeline.py`
- **输出结果**: `/data/lizhiying/Video-Editing/pvtt-main/baseline/compositional-flux-ti2v/pipeline_output/new_video_edit/`
