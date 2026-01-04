# FlowAlign + Image Conditioning 技术方案

## 背景

Baseline 实验发现一个关键问题：当前 FlowAlign 只使用 **文本 prompt** 进行编辑，没有使用 **产品图片** 作为输入。这不符合 PVTT 任务定义。

**PVTT 任务定义**：
- 输入：Template Video + **New Product Image(s)**
- 输出：保持模板风格的新产品视频

**当前 Baseline**：
- 输入：Template Video + Text Prompt
- 缺失：产品图片条件

## 代码分析

### 1. WAN Transformer 已支持 Image Embedding

`transformer_wan.py:409-417`:
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,  # 已支持!
    ...
)
```

### 2. T2V vs I2V 模型差异

| 模型 | `image_dim` | `image_embedder` | 支持图片输入 |
|------|------------|------------------|-------------|
| WAN T2V | `None` | 无 | 否 |
| WAN I2V | `1280` | 有 | 是 |

**关键发现**：必须使用 **I2V 模型** 作为基础，因为 T2V 模型没有 `image_embedder` 层。

### 3. I2V Pipeline 的 Image 处理流程

`pipeline_wan_i2v.py:225-233`:
```python
def encode_image(self, image, device):
    image = self.image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = self.image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]  # Shape: (1, 257, 1280)
```

组件：
- `CLIPVisionModel` - CLIP 图像编码器
- `CLIPImageProcessor` - 图像预处理器

---

## 技术方案

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                   FlowAlign + Image Conditioning                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入:                                                           │
│    ├── Template Video                                           │
│    ├── Source Prompt                                            │
│    ├── Target Prompt                                            │
│    └── Target Product Image ← NEW!                              │
│                                                                  │
│  Pipeline:                                                       │
│    ┌──────────────────┐    ┌──────────────────┐                 │
│    │  Template Video  │    │  Product Image   │                 │
│    └────────┬─────────┘    └────────┬─────────┘                 │
│             │                       │                            │
│             ▼                       ▼                            │
│    ┌──────────────────┐    ┌──────────────────┐                 │
│    │   VAE Encoder    │    │  CLIP Encoder    │                 │
│    └────────┬─────────┘    └────────┬─────────┘                 │
│             │                       │                            │
│             ▼                       ▼                            │
│    ┌──────────────────┐    ┌──────────────────┐                 │
│    │  Video Latents   │    │  Image Embeds    │                 │
│    │   X0_src         │    │  (257, 1280)     │                 │
│    └────────┬─────────┘    └────────┬─────────┘                 │
│             │                       │                            │
│             └───────────┬───────────┘                            │
│                         │                                        │
│                         ▼                                        │
│            ┌────────────────────────┐                            │
│            │   WAN Transformer      │                            │
│            │  (I2V 14B Model)       │                            │
│            │                        │                            │
│            │  encoder_hidden_states │                            │
│            │  = text_embeds         │                            │
│            │                        │                            │
│            │  encoder_hidden_states │                            │
│            │  _image = image_embeds │                            │
│            └───────────┬────────────┘                            │
│                        │                                         │
│                        ▼                                         │
│            ┌────────────────────────┐                            │
│            │   FlowAlign Sampling   │                            │
│            │   (Inversion-Free)     │                            │
│            └───────────┬────────────┘                            │
│                        │                                         │
│                        ▼                                         │
│            ┌────────────────────────┐                            │
│            │   Output Video         │                            │
│            └────────────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### FlowAlign Batch 结构

FlowAlign 使用 batch_size=3 进行推理：

| Index | 用途 | text_embeds | image_embeds |
|-------|------|------------|--------------|
| 0 | Source (for vq) | source_prompt | zeros |
| 1 | Source (for vp) | source_prompt | zeros |
| 2 | Target (for vp) | target_prompt | **target_image** |

---

## 实现步骤

### Step 1: 修改 Pipeline 初始化

```python
# pipeline_wan.py

from transformers import CLIPVisionModel, CLIPImageProcessor

class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    def __init__(
        self,
        tokenizer,
        text_encoder,
        transformer,
        vae,
        scheduler,
        image_encoder: CLIPVisionModel = None,      # 新增
        image_processor: CLIPImageProcessor = None, # 新增
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,      # 新增
            image_processor=image_processor,  # 新增
        )
```

### Step 2: 添加 encode_image 方法

```python
def encode_image(self, image, device=None):
    """
    Encode product image using CLIP vision encoder.

    Args:
        image: PIL Image or tensor
        device: target device

    Returns:
        image_embeds: (batch, 257, 1280)
    """
    device = device or self._execution_device
    image = self.image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = self.image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]
```

### Step 3: 修改 flowalign 方法签名

```python
def flowalign(
    self,
    source_video,
    source_prompt,
    target_prompt,
    target_image=None,  # 新增: 目标产品图片
    negative_prompt=None,
    height=480,
    width=832,
    ...
):
```

### Step 4: 在 flowalign 中编码图片

```python
# 在 flowalign 方法开始处
if target_image is not None and self.image_encoder is not None:
    target_image_embeds = self.encode_image(target_image, device)
else:
    target_image_embeds = None
```

### Step 5: 修改 transformer 调用

```python
# 原代码 (line 1486-1492)
concat_flow_pred = self.transformer(
    hidden_states=concat_latent_model_input,
    timestep=timestep,
    encoder_hidden_states=concat_prompt_embeds,
    attention_kwargs=attention_kwargs,
    return_dict=False,
)[0]

# 修改后
if target_image_embeds is not None:
    # 构建 batch image embeddings
    # Source (index 0, 1) 不使用 image
    # Target (index 2) 使用 target_image
    zero_image_embeds = torch.zeros_like(target_image_embeds)
    concat_image_embeds = torch.cat([
        zero_image_embeds,      # Source for vq
        zero_image_embeds,      # Source for vp
        target_image_embeds,    # Target for vp
    ], dim=0)
else:
    concat_image_embeds = None

concat_flow_pred = self.transformer(
    hidden_states=concat_latent_model_input,
    timestep=timestep,
    encoder_hidden_states=concat_prompt_embeds,
    encoder_hidden_states_image=concat_image_embeds,  # 新增
    attention_kwargs=attention_kwargs,
    return_dict=False,
)[0]
```

---

## 模型加载

### 使用 I2V 模型

```python
from diffusers import WanPipeline, AutoencoderKLWan
from transformers import CLIPVisionModel, CLIPImageProcessor

model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# 加载 image encoder
image_encoder = CLIPVisionModel.from_pretrained(
    model_id,
    subfolder="image_encoder",
    torch_dtype=torch.float32
)
image_processor = CLIPImageProcessor.from_pretrained(
    model_id,
    subfolder="image_processor"
)

# 加载 pipeline
pipe = WanPipeline.from_pretrained(
    model_id,
    image_encoder=image_encoder,
    image_processor=image_processor,
    torch_dtype=torch.bfloat16
)
```

---

## 实验配置

### 新的 YAML 配置格式

```yaml
video:
    video_path: ./videos/bracelet.mp4
    source_prompt: Two personalized couple bracelets...
    target_prompt: A gold charm necklace...
    source_blend: bracelet
    target_blend: necklace
    target_image: ./images/necklace.jpg  # 新增

training-free-type:
    flag_flowedit: False
    flag_flowalign: True

flowalign:
    strength: 0.7
    target_guidance_scale: 19.5
    flag_attnmask: True
    zeta_scale: 1e-3
    save_video: ./results/pvtt/output.mp4
```

---

## 预期效果

| 方面 | 当前 Baseline | 添加 Image 后 |
|------|--------------|--------------|
| 产品外观 | 由 text prompt 生成 | 由产品图片引导 |
| 细节保真度 | 低（模型想象） | 高（基于真实图片） |
| 颜色/材质 | 可能不准确 | 准确匹配图片 |
| 形状 | 语义级别 | 更精确 |

---

## 下一步

1. [ ] 下载 I2V 模型到 5090 机器
2. [ ] 实现代码修改
3. [ ] 准备测试图片（项链产品图）
4. [ ] 运行实验对比
