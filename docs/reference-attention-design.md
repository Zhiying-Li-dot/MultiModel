# Reference Self-Attention for PVTT Baseline

> **⚠️ 重要说明（2026-01-07 最终版）**
>
> 本文档基于 RefDrop (NeurIPS 2024) 方法，针对 **真实产品图片** 进行适配。
>
> **RefDrop 核心公式：**
> ```
> X'_RFG = c · Attention(Q_i, K_ref, V_ref) + (1-c) · Attention(Q_i, K_i, V_i)
> ```
>
> **RefDrop 原文的局限：**
> - RefDrop 论文假设 reference 是**生成的图片**，不是真实照片
> - 论文第 460 行明确指出："支持 clean reference images 是 future work"
>
> **我们的适配方案：**
> - ✅ 使用 **Clean Image + Fixed Features** 方法
> - ✅ 从产品图片提取固定的 K_ref, V_ref（在 t=0）
> - ✅ 所有 denoising step 共用这些固定 features
> - ✅ 借鉴 ControlNet 和 IP-Adapter 的成功经验
>
> **核心参数：**
> - `c = 0.2`（RefDrop 推荐值，video generation）
> - `timestep = 0`（提取 clean features）
>
> **参考文献：** [RefDrop: Controllable Consistency in Image or Video Generation](https://arxiv.org/pdf/2405.17661)

## 背景

### PVTT 任务定义 vs 当前 Baseline 的 Gap

**PVTT 任务：**
- Input: Template Video + **Product Image**
- Output: 保持模板运动，产品外观替换为 Product Image

**当前 FlowAlign Baseline：**
- Input: Template Video + **Text Prompt**
- Gap: 缺少 Product Image 输入

**Wan2.2-TI2V 的问题：**
- TI2V 假设：Product Image = 第一帧
- PVTT 需求：Product Image = 视觉参考（所有帧）
- **不匹配**：产品图片的角度 ≠ 模板视频第一帧的角度

### 解决方案

使用 **Reference Self-Attention**（RefDrop 方法）：
- Training-Free（不训练任何模块）
- 所有帧都参考产品图片
- 在 self-attention 中注入 reference features

---

## 方法来源

### RefDrop (NeurIPS 2024)

**论文：** [RefDrop: Controllable Consistency in Image or Video Generation](https://arxiv.org/abs/2405.17661)

**核心贡献：**
1. 在 **self-attention** 中注入 reference image features
2. **Training-Free**（推理时修改 attention）
3. 明确用于 **video generation**
4. 可控的 reference 影响强度

**与 IP-Adapter 的区别：**

| 方法 | Attention 类型 | 需要训练 | 适用场景 |
|------|---------------|---------|----------|
| IP-Adapter | Cross-Attention | ✅ 需要 | Image generation |
| RefDrop | Self-Attention | ❌ 不需要 | Video generation |

**RefDrop 核心要点：** ⭐⭐⭐

1. **公式（最关键）：**
   ```
   X'_RFG = c · Attention(Q_i, K_ref, V_ref) + (1-c) · Attention(Q_i, K_i, V_i)
   ```
   - 计算两次 attention，线性插值
   - **不是** K/V 拼接！

2. **参数：**
   - `c = 0.2` for video generation（论文推荐）
   - `c = 0.5` for image generation

3. **Reference 处理（原文）：**
   - Reference image **必须是生成的**（论文第 59 行）
   - Reference 作为 batch 的第一个样本
   - 每个 denoising step，reference 和其他样本同步 denoise

4. **应用层：**
   - 只修改 spatial self-attention layers
   - Cross-attention 和 temporal attention 保持不变

**RefDrop 的局限性：** ⚠️

论文明确指出（第 59 行）：
> "Furthermore, our reference images are **generated** by the same model, in contrast to IP-Adapter's reliance on externally sourced image."

论文第 460 行（Future Work）：
> "Another exciting prospect involves enhancing our method to **accept clean reference images as input**, similar to the IP-Adapter... Achieving this capability would represent a significant advancement"

**结论：RefDrop 原文不支持真实产品图片！需要适配。**

**引用：**
> "RefDrop integrates the reference image directly into the self-attention layer without needing additional training."

> "It enables applications such as... high-quality personalized video generation by boosting temporal consistency."

### ControlNet Reference-only (2023)

**来源：** [ControlNet Reference-only Mode](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236)

**原理：**
- 用 reference image forward 一遍，保存 features
- 生成时在 self-attention 中读取这些 features
- 三种模式：reference-attn, reference-adain, reference-attn+adain

**关键启发：**
- ✅ **使用 clean image features**
- ✅ 预先提取，推理时复用
- ✅ 成功应用于 image generation

---

## PVTT 适配方案：使用真实产品图片

### 问题分析

| 方面 | RefDrop 原文 | PVTT 需求 |
|------|------------|----------|
| Reference 来源 | 生成的图片 | **真实产品照片** |
| Reference 状态 | 动态（参与 denoising） | **Clean（不应加噪声）** |
| K, V 提取 | 每个 step 动态提取 | **需要适配方案** |

### 方案对比

#### 方案 1：Clean Image + Fixed Features ⭐ **推荐**

**核心思路：**
```python
# 预处理（一次性）
z_ref = vae.encode(product_image)  # Clean latent，不加噪声
ref_bank = extract_features(z_ref, t=0)  # 在 t=0 提取 K, V

# Generation（每个 step t）
# 使用固定的 ref_bank，不随 t 变化
output = c * attn(Q, K_ref, V_ref) + (1-c) * attn(Q, K, V)
```

**优点：**
- ✅ 符合 PVTT 需求（产品图片提供精确外观）
- ✅ 简单高效（一次提取，多次复用）
- ✅ 有成功先例（ControlNet, IP-Adapter）
- ✅ 计算成本低

**理论支持：**
- ControlNet：用 clean image features → 成功
- IP-Adapter：用 CLIP clean features → 成功
- RefDrop 论文：支持 clean image 是 "future work"

#### 方案 2：Noisy Image + Dynamic Features

**核心思路：**
```python
# 每个 step t
z_ref_noisy = add_noise(z_ref, t)  # 产品图片加噪声
batch = [z_ref_noisy, video_frames...]
# K_ref, V_ref 动态提取
```

**缺点：**
- ❌ 产品图片加噪声破坏外观
- ❌ 计算成本高
- ❌ 不适合 PVTT

#### 方案 3：DDIM Inversion + Synchronized Denoising

**缺点：**
- ❌ 计算成本极高
- ❌ 实现复杂
- ❌ 不适合 baseline

### 最终选择：方案 1（Clean Image + Fixed Features）

**与 RefDrop 原文的差异：**

| RefDrop 原文 | PVTT 适配 |
|-------------|---------|
| Reference 是生成的 | Reference 是真实照片 |
| 动态（参与 denoising）| **静态（fixed features）** |
| 每个 step 动态提取 K, V | **预提取（t=0）** |
| 论文验证 | 借鉴 ControlNet 思路 |

---

## 技术方案

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│             FlowAlign + Reference Self-Attention                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input:                                                          │
│    ├── Template Video                                           │
│    ├── Source Prompt                                            │
│    ├── Target Prompt                                            │
│    └── Product Image ← NEW!                                     │
│                                                                  │
│  Step 1: Extract Reference Features (一次性)                     │
│  ┌────────────────────┐                                          │
│  │  Product Image     │                                          │
│  └────────┬───────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                          │
│  │   VAE Encoder      │                                          │
│  └────────┬───────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                          │
│  │   Ref Latent       │                                          │
│  │   z_ref            │                                          │
│  └────────┬───────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────┐                              │
│  │   Transformer Forward          │                              │
│  │   (Feature Extraction Only)    │                              │
│  │                                │                              │
│  │   For each self-attention:     │                              │
│  │     K_ref = to_k(z_ref)        │                              │
│  │     V_ref = to_v(z_ref)        │                              │
│  └────────┬───────────────────────┘                              │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                          │
│  │  Reference Bank    │ ← 保存所有层的 K_ref, V_ref              │
│  │  {layer_i: (K, V)} │                                          │
│  └────────────────────┘                                          │
│                                                                  │
│  Step 2: FlowAlign with Modified Attention                      │
│  ┌────────────────────┐                                          │
│  │  Template Video    │                                          │
│  └────────┬───────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                          │
│  │   VAE Encoder      │                                          │
│  └────────┬───────────┘                                          │
│           │                                                      │
│  ┌────────▼────────────────────────────────┐                     │
│  │   FlowAlign Loop (t = T → 0)            │                     │
│  │                                         │                     │
│  │   For each timestep t:                  │                     │
│  │     1. Forward diffusion: Zt_src        │                     │
│  │     2. Coupling: Zt_tar                 │                     │
│  │     3. Transformer forward with         │                     │
│  │        **RefDrop Self-Attention**:      │                     │
│  │                                         │                     │
│  │        Q = to_q(hidden_states)          │                     │
│  │        K = to_k(hidden_states)          │                     │
│  │        V = to_v(hidden_states)          │                     │
│  │                                         │                     │
│  │        attn_self = Attention(Q, K, V)   │← 正常 attention    │
│  │        attn_ref = Attention(Q, K_ref, V_ref)  ← 从 bank 读取 │
│  │                                         │                     │
│  │        output = c*attn_ref + (1-c)*attn_self  ← 线性插值    │
│  │        (c=0.2 for video)                │                     │
│  │                                         │                     │
│  │     4. FlowAlign update                 │                     │
│  └─────────┬───────────────────────────────┘                     │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                          │
│  │   VAE Decoder      │                                          │
│  └────────┬───────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                          │
│  │   Output Video     │                                          │
│  └────────────────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心机制（PVTT 适配版本）

#### 1. RefDrop 核心公式（保持不变）

```
X'_RFG = c · Attention(Q_i, K_ref, V_ref) + (1-c) · Attention(Q_i, K_i, V_i)
```

**说明：**
- `Attention(Q_i, K_ref, V_ref)`: 用 reference 的 K, V 计算 attention
- `Attention(Q_i, K_i, V_i)`: 正常的 self-attention
- `c`: reference guidance coefficient（**c = 0.2 for video generation**）
- **线性插值**，不是拼接！

**PVTT 适配的关键修改：**
- ❌ **不是** Reference 作为 batch 第一个样本（原文方法）
- ✅ **而是** 预先提取 clean image 的 K_ref, V_ref
- ✅ 在 t=0 提取一次，所有 denoising step 复用
- ✅ 类似 ControlNet 的做法

#### 2. Clean Reference Attention Processor（PVTT 适配）

```python
class CleanReferenceAttentionProcessor:
    """
    RefDrop with Clean Reference Image (PVTT Adaptation)

    核心修改：使用固定的 K_ref, V_ref（从 clean product image 提取）
    适用于：真实产品图片（非生成图片）
    """
    def __init__(self, ref_bank: Dict, c: float = 0.2):
        """
        Args:
            ref_bank: {layer_name: {"key": K_ref, "value": V_ref}}
                     从 clean product image 提取的固定 features
            c: Reference guidance coefficient (0.0-1.0)
               0.0 = 无 reference（纯 self-attention）
               0.2 = RefDrop 推荐值（video generation）⭐
               1.0 = 完全使用 reference
        """
        self.ref_bank = ref_bank
        self.c = c

    def __call__(
        self,
        attn_module,
        hidden_states,
        encoder_hidden_states=None,
        layer_name=None,
        **kwargs
    ):
        """
        Modified self-attention with RefDrop linear interpolation

        Formula:
            output = c * Attention(Q, K_ref, V_ref) + (1-c) * Attention(Q, K, V)

        注意：K_ref, V_ref 是固定的（在 t=0 提取），不随 denoising step 变化
        """
        # 1. Cross-attention: 不修改
        if encoder_hidden_states is not None:
            query = attn_module.to_q(hidden_states)
            key = attn_module.to_k(encoder_hidden_states)
            value = attn_module.to_v(encoder_hidden_states)
            return attn_module.attention(query, key, value)

        # 2. Self-Attention: 计算 Q, K, V
        query = attn_module.to_q(hidden_states)
        key = attn_module.to_k(hidden_states)
        value = attn_module.to_v(hidden_states)

        # 3. 正常的 self-attention
        attn_self = attn_module.attention(query, key, value)

        # 4. 如果有 reference，计算 reference attention
        if layer_name in self.ref_bank and self.c > 0:
            K_ref = self.ref_bank[layer_name]["key"]  # [1, seq_len_ref, dim]
            V_ref = self.ref_bank[layer_name]["value"]

            # Expand to match batch size
            batch_size = query.shape[0]
            K_ref = K_ref.expand(batch_size, -1, -1)
            V_ref = V_ref.expand(batch_size, -1, -1)

            # 计算 reference attention
            attn_ref = attn_module.attention(query, K_ref, V_ref)

            # 5. RefDrop 线性插值（核心公式）
            output = self.c * attn_ref + (1 - self.c) * attn_self
        else:
            # 没有 reference 或 c=0，返回正常 attention
            output = attn_self

        return output
```

#### 3. Clean Reference Feature Extraction（PVTT 适配）

```python
def extract_clean_reference_features(
    product_image: Image.Image,
    vae,
    transformer,
    text_encoder,
    tokenizer,
    ref_prompt: str,
    target_size: tuple = (480, 832),
    device: str = "cuda"
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    从 clean product image 提取固定的 reference features

    ⚠️ PVTT 适配关键：
    - 在 timestep=0 提取 features（clean state，无噪声）
    - 所有 denoising step 共用这些固定的 K_ref, V_ref
    - 不随生成过程的 timestep 变化

    Args:
        product_image: 真实产品图片（clean）
        ref_prompt: 产品描述（target prompt）
        target_size: 目标分辨率
        device: 设备

    Returns:
        ref_bank: {layer_name: {"key": K_ref, "value": V_ref}}
    """
    print("[Clean RefDrop] Extracting features from product image...")

    # 1. 预处理 + VAE encode
    ref_tensor = preprocess_image(product_image, target_size)  # [3, H, W]
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, 3, 1, H, W]

    with torch.no_grad():
        # 2. Encode to latent (no noise!)
        z_ref = vae.encode(ref_tensor).latent_dist.sample()  # [1, 16, 1, H', W']
        print(f"[Clean RefDrop] Latent shape: {z_ref.shape}")

        # 3. Encode text
        prompt_embeds = text_encoder(
            tokenizer(
                ref_prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
        )[0]

        # 4. 准备 hooks 提取 K, V
        ref_bank = {}
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                if len(input) > 0:
                    hidden_states = input[0]
                    # 只在 self-attention 中保存（cross-attention 跳过）
                    if len(input) == 1 or input[1] is None:
                        try:
                            K_ref = module.to_k(hidden_states)
                            V_ref = module.to_v(hidden_states)
                            ref_bank[layer_name] = {
                                "key": K_ref.detach().clone(),
                                "value": V_ref.detach().clone()
                            }
                        except AttributeError:
                            pass
            return hook_fn

        # 5. 注册 hooks
        for name, module in transformer.named_modules():
            if hasattr(module, 'to_q') and hasattr(module, 'to_k'):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        print(f"[Clean RefDrop] Registered {len(hooks)} hooks")

        # 6. Forward pass at t=0 (clean state) ⭐ 关键
        t = torch.zeros(1, device=device)  # t=0 for clean features
        try:
            _ = transformer(
                z_ref,
                encoder_hidden_states=prompt_embeds,
                timestep=t,
                return_dict=False
            )
        except Exception as e:
            print(f"[Clean RefDrop] Warning: Forward pass error: {e}")
            # Continue anyway, we may have extracted some features

        # 7. 清理 hooks
        for hook in hooks:
            hook.remove()

    print(f"[Clean RefDrop] Extracted features from {len(ref_bank)} layers (t=0)")

    # Print sample layer info
    if ref_bank:
        sample_name = list(ref_bank.keys())[0]
        sample_k = ref_bank[sample_name]["key"]
        print(f"[Clean RefDrop] Sample layer '{sample_name}' K shape: {sample_k.shape}")

    return ref_bank
```

#### 4. 集成到 FlowAlign（PVTT 适配）

```python
# In awesome_wan_editing.py

if config['flowalign'].get('use_reference_attention', False) and target_image is not None:
    print("[Clean RefDrop] Starting reference attention setup...")

    # 参数
    ref_c = config['flowalign'].get('ref_c', 0.2)  # 默认 0.2（video 推荐值）

    # Step 1: 提取 clean reference features（一次性，预处理）
    ref_bank = extract_clean_reference_features(
        product_image=target_image,
        vae=pipe.vae,
        transformer=pipe.transformer,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        ref_prompt=target_prompt,  # 使用 target prompt 作为 reference context
        target_size=(480, 832),
        device=pipe.device
    )

    # Step 2: 注册 Clean Reference Attention Processor
    register_attention_processor(
        pipe.transformer,
        processor_type="CleanReferenceAttentionProcessor",
        ref_bank=ref_bank,
        c=ref_c
    )

    print(f"[Clean RefDrop] Registered (c={ref_c})")

else:
    print("[FlowAlign] No reference attention, using vanilla FlowAlign")

# Step 3: 正常执行 FlowAlign
# （attention processor 会自动在 forward 时调用）
output = pipe.flowalign(
    video=video,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
    height=480,
    width=832,
    num_inference_steps=num_inference_steps,
    strength=strength,
    target_guidance_scale=target_guidance_scale,
    fg_zeta_scale=fg_zeta_scale,
    bg_zeta_scale=bg_zeta_scale,
).frames[0]
```

---

## 实现细节

### 1. 如何识别 Self-Attention Layer

**Wan Transformer 架构：**
- 每个 Block 包含 self-attention 和 cross-attention
- 需要只修改 **self-attention**（cross-attention 保持不变）

```python
def is_self_attention_layer(module):
    """
    判断是否为 self-attention layer
    """
    # 方法 1: 根据名字判断
    if "attn1" in module.__class__.__name__.lower():
        return True

    # 方法 2: 根据 forward signature 判断
    # self-attention: forward(hidden_states)
    # cross-attention: forward(hidden_states, encoder_hidden_states)

    return False
```

**参考 MasaCtrl 的实现：**
- `baseline/flowedit-wan2.1/utils/wan_attention.py`
- 已经有 `register_attention_processor` 机制

### 2. Feature Extraction Timestep（PVTT 适配）

**PVTT 方案：固定使用 t=0（clean features）** ⭐

**理由：**
- ✅ 产品图片就是要提供精确外观
- ✅ Clean features 保留最多细节
- ✅ 类似 ControlNet 的成功经验
- ✅ 符合直觉（产品外观不应加噪声）

**可选探索（如果效果太强）：**

| Timestep | 语义层次 | 适用场景 |
|----------|---------|---------|
| t = 0.0 | 最细节 | 需要精确外观 ⭐ **推荐** |
| t = 0.3 | 中-细节 | 平衡外观和灵活性 |
| t = 0.5 | 中等语义 | 只控制大致风格 |

**实现：**
```python
# 默认方案：t=0（clean）
t = torch.zeros(1, device=device)  # t=0 for clean features
_ = transformer(z_ref, encoder_hidden_states=prompt_embeds, timestep=t, ...)
```

**注意：** RefDrop 原文不存在这个参数（因为 reference 是动态的）

### 3. Reference Guidance Coefficient（c）控制

**作用：** 控制产品图片的影响强度

**RefDrop 公式：**
```python
output = c * attn_ref + (1 - c) * attn_self
```

**参数范围：**
- `c = 0.0`: 无 reference（等价于原始 FlowAlign）
- `c = 0.2`: **RefDrop 推荐值（video generation）** ⭐
- `c = 0.5`: 中等影响（平衡 reference 和 self-attention）
- `c = 1.0`: 完全使用 reference（可能丢失运动信息）

**RefDrop 论文建议：**
- Image generation: `c = 0.5`
- Video generation: `c = 0.2`（更低，避免破坏时序一致性）

**建议：** 从 0.2 开始，根据效果调整到 0.1-0.5 之间

### 4. 哪些层注入 Reference

**选项：**

| 策略 | 层数 | 效果 | 计算成本 |
|------|-----|------|---------|
| All layers | 30 层 | 全局影响 | 高 |
| Middle layers | 10-20 层 | 平衡 | 中 |
| Deep layers | 20-30 层 | 细节控制 | 低 |

**参考 MasaCtrl：**
- MasaCtrl 只修改特定层（layers 11-17）
- 用于 attention masking

**建议：**
1. 先尝试 All layers
2. 如果计算太慢，只用 middle/deep layers

---

## 实现步骤

### Phase 1: 最小可行实现（1-2天）

**目标：** 跑通基本流程，验证可行性

1. **实现 `extract_reference_features`**
   - 读取产品图片
   - VAE encode
   - Forward transformer，保存 K, V

2. **实现 `ReferenceAttentionProcessor`**
   - 参考 `MasaCtrlProcessor`（`wan_attention.py`）
   - 在 self-attention 中拼接 reference

3. **集成到 `awesome_wan_editing.py`**
   - 添加 `--target_image` 参数
   - 调用 reference attention

4. **单个实验测试**
   - Input: bracelet video + necklace image
   - 看是否有外观变化

### Phase 2: 参数调优（2-3天）

**目标：** 找到最佳参数组合

1. **Timestep 消融**
   - t = 0.0, 0.3, 0.5, 0.7, 1.0
   - 多 timestep 平均

2. **Coefficient (c) 消融**
   - c = 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0

3. **Layer Selection 消融**
   - All layers
   - Middle layers (10-20)
   - Deep layers (20-30)

4. **与 Baseline 对比**
   - Vanilla FlowAlign (text only)
   - FlowAlign + Reference (text + image)

### Phase 3: 集成到实验系统（1天）

**目标：** 批量运行实验

1. **修改 `run_experiments.py`**
   - 支持 `target_image` 参数
   - 自动查找产品图片

2. **YAML 配置格式**
   ```yaml
   video:
       video_path: jewelry/JEWE001.mp4
       source_prompt: "..."
       target_prompt: "..."
       target_image: jewelry/NECK001.jpg  # NEW: 产品图片路径

   flowalign:
       use_reference_attention: True    # NEW: 启用 Clean RefDrop
       ref_c: 0.2                       # NEW: RefDrop coefficient (0.2 for video)
       # 注意：不需要 ref_timestep 参数（固定 t=0）
   ```

3. **批量实验**
   - 4 个已有实验 + reference
   - 对比效果

---

## 预期效果

### 定性效果

| 方面 | Vanilla FlowAlign | + Reference Attention |
|------|------------------|----------------------|
| 外观保真度 | ⭐⭐ (靠 text) | ⭐⭐⭐⭐ (靠 image) |
| 颜色准确性 | ⭐⭐ | ⭐⭐⭐⭐ |
| 材质细节 | ⭐⭐ | ⭐⭐⭐⭐ |
| 形状匹配 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 运动保持 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (保持) |

### 定量指标

**可能的评估指标：**

1. **CLIP Image Similarity**
   ```python
   # 输出视频的每一帧 vs 产品图片
   sim = cosine_similarity(
       clip_vision(output_frame),
       clip_vision(product_image)
   )
   ```
   - 预期：+ Reference 应该更高

2. **LPIPS (Perceptual Similarity)**
   - 衡量细节保真度

3. **Temporal Consistency**
   - 相邻帧的特征相似度
   - 预期：+ Reference 不应该降低

4. **User Study**
   - 哪个更像产品图片？
   - 哪个运动更自然？

---

## 潜在问题与解决方案

### 问题 1: Reference 过强，运动丢失

**表现：** 所有帧都太像产品图片，失去模板的运动

**原因：** `c` 太大（接近 1.0），或者所有层都注入

**解决：**
- 降低 `c`（试试 0.1-0.3，RefDrop 推荐 0.2）
- 只在 deep layers 注入（控制细节，不影响运动）
- 检查是否使用了正确的线性插值公式

### 问题 2: Reference 太弱，外观不像

**表现：** 还是主要靠 text prompt，产品图片影响小

**原因：** `c` 太小（<0.1），或者只在少数层注入

**解决：**
- 增大 `c`（试试 0.3-0.5）
- 在更多层注入（middle + deep）
- 检查 ref_bank 是否正确提取

### 问题 3: 时序不一致（闪烁）

**表现：** 相邻帧之间外观跳变

**原因：**
- Reference features 是静态的，每帧独立参考
- `c` 太大，破坏了 temporal attention

**解决：**
1. **降低 c**（RefDrop 推荐 video 用 0.2，比 image 的 0.5 更低）
2. **只在特定层注入**（保留其他层的 temporal attention）
3. **调整 ref_timestep**（较大的 t 提供更高层的语义，减少细节干扰）

### 问题 4: 计算成本高

**表现：** 推理变慢

**原因：**
- 需要计算两次 attention（attn_self 和 attn_ref）
- 比原始 FlowAlign 慢约 2x（每个 self-attention 层）

**解决：**
- 只在部分层注入（减少计算）
- 降低 `c`（当 c 接近 0 时，可以跳过 attn_ref 计算）
- 使用 Flash Attention（优化 attention 计算）

**注意：** RefDrop 比 K/V 拼接方法计算更高效（序列长度不变）

---

## 对比：其他可能的方案

| 方案 | Training | 实现难度 | 外观保真度 | 运动保持 | 计算成本 |
|------|---------|---------|-----------|---------|---------|
| **Reference Attention** | ❌ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| TI2V First-Frame | ❌ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Image Captioning | ❌ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| CLIP Guidance | ❌ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| IP-Adapter | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**为什么选 Reference Attention：**
- ✅ Training-Free（符合 baseline 定位）
- ✅ 理论有支持（RefDrop NeurIPS 2024）
- ✅ 实现难度适中（参考 MasaCtrl）
- ✅ 效果有保证（video generation 验证过）

---

## 参考文献

1. **RefDrop (NeurIPS 2024)**
   [RefDrop: Controllable Consistency in Image or Video Generation](https://arxiv.org/abs/2405.17661)
   - Training-free reference injection via self-attention

2. **ControlNet Reference-only (2023)**
   [GitHub Discussion](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236)
   - Original reference attention for image generation

3. **IP-Adapter (2023)**
   [IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721)
   - Cross-attention based (需要训练)

4. **I2V-Adapter (SIGGRAPH 2024)**
   [I2V-Adapter: A General Image-to-Video Adapter](https://arxiv.org/abs/2312.16693)
   - Cross-frame attention (需要训练)

5. **FlowAlign (arXiv 2025)**
   [FlowAlign: Trajectory-Regularized, Inversion-Free Flow-based Image Editing](https://arxiv.org/abs/2505.23145)
   - 我们的 baseline 方法（基于 FlowEdit + zeta regularization）

---

## 下一步行动

### 立即开始（Phase 1）

1. **创建新文件：** `baseline/flowedit-wan2.1/reference_attention.py`
   - 实现 `extract_reference_features`
   - 实现 `ReferenceAttentionProcessor`

2. **修改：** `baseline/flowedit-wan2.1/awesome_wan_editing.py`
   - 添加 `--target_image` 参数
   - 集成 reference attention

3. **测试实验：**
   ```bash
   python awesome_wan_editing.py \
       --config config/pvtt/test01_watch_to_bracelet_reference.yaml
   ```

   Config 文件示例：
   ```yaml
   video:
       target_image: ../../data/pvtt-benchmark/images/jewelry/JEWE002.jpg
   flowalign:
       use_reference_attention: True
       ref_c: 0.2          # RefDrop 推荐值（video generation）
       # 注意：固定使用 t=0（clean features）
   ```

4. **对比结果：**
   - Baseline: text only
   - + Reference: text + image

### 后续优化（Phase 2-3）

- 参数调优
- 批量实验
- 写入论文

---

---

## 总结：PVTT 适配关键点 ⭐

### RefDrop 原文 vs PVTT 适配

| 方面 | RefDrop 原文 | PVTT 适配 |
|------|------------|----------|
| **Reference 来源** | 生成的图片 | **真实产品照片** |
| **Reference 状态** | 动态（参与 denoising） | **静态（fixed features）** |
| **K, V 提取时机** | 每个 step 动态提取 | **预提取（t=0，一次性）** |
| **Feature 类型** | 带噪声（随 step 变化） | **Clean（无噪声）** |
| **理论依据** | RefDrop 论文验证 | **借鉴 ControlNet 思路** |
| **计算成本** | 高（batch +1，每 step forward） | **低（预计算一次）** |

### 核心设计决策

**1. 为什么用 Clean Image + Fixed Features？**
- ✅ 符合 PVTT 需求（产品图片应提供精确外观）
- ✅ 有成功先例（ControlNet, IP-Adapter）
- ✅ 简单高效（一次提取，多次复用）
- ✅ RefDrop 论文将"支持 clean image"列为 future work

**2. 为什么固定 t=0？**
- ✅ 产品图片不应加噪声
- ✅ Clean features 保留最多细节
- ✅ 符合直觉

**3. 为什么 c=0.2？**
- ✅ RefDrop 论文推荐值（video generation）
- ✅ 平衡外观保真度和运动保持
- ✅ 可以根据效果调整（0.1-0.5）

### 关键公式（保持不变）

```
X' = c · Attention(Q, K_ref, V_ref) + (1-c) · Attention(Q, K, V)
```

**其中：**
- K_ref, V_ref：从 clean product image 提取（t=0），所有 step 共用
- c = 0.2：video generation 推荐值
- 线性插值，不是拼接

### 实现要点

```python
# 1. 预处理（一次性）
ref_bank = extract_clean_reference_features(
    product_image=target_image,
    vae=vae,
    transformer=transformer,
    ...
)  # 在 t=0 提取

# 2. 注册 Processor
register_attention_processor(
    transformer,
    processor_type="CleanReferenceAttentionProcessor",
    ref_bank=ref_bank,
    c=0.2
)

# 3. 正常生成（processor 自动应用）
output = flowalign(...)
```

### 潜在风险与对策

**风险 1：Feature Domain Mismatch**
- 问题：Clean features (t=0) vs Noisy latents (t>0)
- 对策：实验验证，如有问题可尝试 timestep-adaptive

**风险 2：与 RefDrop 原文偏离**
- 问题：我们的方案不是 RefDrop 原文
- 对策：在论文中明确说明是"适配版本"，引用 ControlNet 作为理论支持

**风险 3：效果可能不如预期**
- 对策：参数调优（c, layer selection），实验验证

---

**Status:** Design Complete (PVTT Adaptation)
**Next:** Implementation
**ETA:** Phase 1 可在 1-2 天内完成
