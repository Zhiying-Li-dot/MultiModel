# PVTT 组合式方法技术方案

> **版本**: v1.0 (2026-01-09)
>
> **背景**: 端到端 Training-Free 方法（RefDrop）效果有限，c 值过大会破坏编辑，过小则引导不足。
>
> **新思路**: 将任务拆解为两阶段，分别用最适合的模型处理。

## 1. 方法概述

### 1.1 核心思路

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PVTT 组合式方法流程                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   输入                     Stage 1                    Stage 2       │
│  ┌─────┐                  ┌─────────┐               ┌─────────┐    │
│  │模板 │ 提取第一帧        │ Flux.2  │  生成目标     │ Wan2.1  │    │
│  │视频 │ ─────────────────▶│  Dev    │─────────────▶│  TI2V   │    │
│  └─────┘                  │  Edit   │  第一帧       │         │    │
│                           └────┬────┘               └────┬────┘    │
│  ┌─────┐                       │                         │         │
│  │产品 │ ──────────────────────┘                         │         │
│  │图片 │                                                 ▼         │
│  └─────┘                                            ┌─────────┐    │
│                                                     │ 目标视频 │    │
│                                                     └─────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 两阶段分工

| 阶段 | 任务 | 模型 | 输入 | 输出 |
|------|------|------|------|------|
| **Stage 1** | 图像编辑 | Flux.2 Dev Edit | 模板第一帧 + 产品图 | 目标第一帧 |
| **Stage 2** | 视频生成 | Wan2.1 TI2V | 目标第一帧 + Prompt | 目标视频 |

### 1.3 对比端到端方法

| 方面 | 端到端 (RefDrop) | 组合式 (Ours) |
|------|------------------|---------------|
| 复杂度 | 单模型，需精细调参 | 两阶段，各司其职 |
| 产品保真度 | 低（c=0.05 才能用） | 高（Flux.2 专长） |
| 运动保持 | 好（FlowEdit 机制） | 依赖 TI2V 质量 |
| 可调试性 | 难（黑盒融合） | 易（中间结果可检查） |

## 2. Stage 1: 图像编辑 (Flux.2 Dev Edit)

### 2.1 模型选择

**Flux.2 Dev Edit** (Black Forest Labs, 2025.11)
- 32B 参数 Rectified Flow Transformer
- 支持多参考图编辑（Multi-Reference Editing）
- 自然语言指令，无需 mask 或 finetuning

### 2.2 输入输出

```yaml
输入:
  image_1: 模板视频第一帧  # 提供场景、光照、构图
  image_2: 目标产品图片    # 提供产品外观
  prompt: |
    Replace the [source_product] in @image_1 with the [target_product] from @image_2.
    Maintain the same lighting, camera angle, background, and composition.
    The new product should fit naturally in the scene.

参数:
  guidance_scale: 2.5-5.0  # 控制 prompt 遵循程度
  num_inference_steps: 28  # 默认值
  image_size: 832x480      # 匹配视频分辨率

输出:
  target_first_frame: 目标视频第一帧
```

### 2.3 Prompt 模板

```python
EDIT_PROMPT_TEMPLATE = """
Replace the {source_object} in @image_1 with the {target_object} from @image_2.

Requirements:
- Keep the exact same camera angle and framing as @image_1
- Preserve the lighting and shadows from @image_1
- Maintain the background and surrounding elements from @image_1
- The {target_object} should match the style, material, and details of @image_2
- Ensure natural integration with proper scale and perspective
"""
```

### 2.4 API 调用示例

```python
import fal_client

def generate_target_first_frame(
    template_first_frame_url: str,
    product_image_url: str,
    source_object: str,
    target_object: str,
) -> str:
    """
    使用 Flux.2 Dev Edit 生成目标视频第一帧。

    Args:
        template_first_frame_url: 模板视频第一帧 URL
        product_image_url: 目标产品图片 URL
        source_object: 源产品描述（如 "bracelets"）
        target_object: 目标产品描述（如 "gold necklace with pearls"）

    Returns:
        生成的目标第一帧 URL
    """
    prompt = EDIT_PROMPT_TEMPLATE.format(
        source_object=source_object,
        target_object=target_object,
    )

    result = fal_client.subscribe(
        "fal-ai/flux-2/edit",
        arguments={
            "prompt": prompt,
            "image_urls": [
                template_first_frame_url,
                product_image_url,
            ],
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "image_size": {"width": 832, "height": 480},
            "output_format": "png",
        },
    )

    return result["images"][0]["url"]
```

### 2.5 预期效果

| 测试用例 | 输入 | 期望输出 |
|---------|------|---------|
| Bracelet → Necklace | 黑银手环场景 + 珍珠项链图 | 紫色丝绸上的珍珠项链 |
| Watch → Bracelet | 手表展示 + 手链图 | 同场景下的手链 |

## 3. Stage 2: 视频生成 (Wan2.1 TI2V)

### 3.1 模型选择

**Wan2.1 TI2V-5B** (Alibaba, 2025)
- Text-Image-to-Video 模型
- 以第一帧为条件生成视频
- 支持 49 帧 @ 16fps

### 3.2 输入输出

```yaml
输入:
  first_frame: Stage 1 输出的目标第一帧
  prompt: 目标产品描述 + 运动描述

参数:
  num_frames: 49
  fps: 16
  guidance_scale: 5.0

输出:
  target_video: 目标产品视频
```

### 3.3 Prompt 设计

TI2V 的 prompt 需要描述：
1. **产品外观** - 与 Stage 1 一致
2. **运动/动态** - 参考模板视频的运动模式

```python
TI2V_PROMPT_TEMPLATE = """
{product_description}

The scene shows {motion_description}.
Camera {camera_motion}.
Professional product photography with {lighting_style} lighting.
"""

# 示例
prompt = TI2V_PROMPT_TEMPLATE.format(
    product_description="A gold chain necklace with white pearl drop pendants and red gemstone accents placed on purple silk fabric",
    motion_description="the jewelry resting elegantly on flowing silk with subtle fabric movement",
    camera_motion="slowly zooms in with gentle movement",
    lighting_style="soft studio",
)
```

### 3.4 运动一致性问题

**挑战**: TI2V 生成的运动可能与模板视频不同

**潜在解决方案**:
1. **Motion Prompt Engineering** - 详细描述运动模式
2. **Motion LoRA** - 如果有类似运动的训练数据
3. **Hybrid 方法** - 结合 FlowAlign 的 attention 机制
4. **后处理** - 使用视频插帧/光流对齐

## 4. 实现计划

### 4.1 Phase 1: 验证 Stage 1 (图像编辑)

```
[ ] 1. 注册 fal.ai 账号，获取 API Key
[ ] 2. 实现 Flux.2 Dev Edit 调用脚本
[ ] 3. 测试 bracelet → necklace 案例
[ ] 4. 评估生成的第一帧质量
    - 产品保真度
    - 场景一致性
    - 光照自然度
```

### 4.2 Phase 2: 验证 Stage 2 (视频生成)

```
[ ] 5. 使用 Stage 1 输出作为 TI2V 输入
[ ] 6. 测试不同 motion prompt 效果
[ ] 7. 对比生成视频与模板视频的运动
```

### 4.3 Phase 3: 端到端 Pipeline

```
[ ] 8. 整合两阶段为完整 pipeline
[ ] 9. 在 PVTT benchmark 上测试
[ ] 10. 与 RefDrop baseline 对比
```

## 5. 评估指标

### 5.1 Stage 1 评估（图像）

| 指标 | 说明 | 工具 |
|------|------|------|
| 产品保真度 | 生成产品与参考图相似度 | CLIP-I, DINO |
| 场景一致性 | 背景/光照保持程度 | LPIPS, SSIM |
| 人工评估 | 自然度、质量 | 用户打分 |

### 5.2 Stage 2 评估（视频）

| 指标 | 说明 | 工具 |
|------|------|------|
| 时序一致性 | 产品外观帧间稳定 | FVD |
| 运动相似度 | 与模板运动的匹配度 | 光流对比 |
| 视频质量 | 整体生成质量 | CLIP-T |

## 6. 风险与备选方案

### 6.1 Stage 1 风险

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| Flux.2 产品替换不精确 | 产品细节丢失 | 尝试 Flux Fill + Redux 组合 |
| 场景变化过大 | 背景不一致 | 使用 inpainting mask 限制编辑区域 |
| API 成本过高 | 批量测试受限 | 本地部署 Flux.2 Dev |

### 6.2 Stage 2 风险

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| TI2V 运动与模板不符 | 无法保持模板风格 | 尝试 Wan2.2 或其他 TI2V |
| 产品外观帧间不稳定 | 闪烁/变形 | 增加 temporal consistency loss |
| 生成质量不足 | 商业不可用 | 结合 FlowAlign 做后处理 |

## 7. 参考资料

- [FLUX.2 Dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [FLUX.2 Multi-Reference Editor on fal.ai](https://fal.ai/models/fal-ai/flux-2/edit)
- [Wan2.1 TI2V 文档](https://huggingface.co/Wan-AI)
- [RefDrop 实验记录](../experiments/README.md)
