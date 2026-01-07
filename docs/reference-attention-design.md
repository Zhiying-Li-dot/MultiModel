# Reference Self-Attention for PVTT Baseline

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

**引用：**
> "RefDrop integrates the reference image directly into the self-attention layer without needing additional training."

> "It enables applications such as... high-quality personalized video generation by boosting temporal consistency."

### ControlNet Reference-only (2023)

**来源：** [ControlNet Reference-only Mode](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236)

**原理：**
- 用 reference image forward 一遍，保存 features
- 生成时在 self-attention 中读取这些 features
- 三种模式：reference-attn, reference-adain, reference-attn+adain

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
│  │        **Modified Self-Attention**:     │                     │
│  │                                         │                     │
│  │        Q = to_q(hidden_states)          │                     │
│  │        K = to_k(hidden_states)          │                     │
│  │        V = to_v(hidden_states)          │                     │
│  │                                         │                     │
│  │        K' = concat([K, K_ref], dim=1)   │← 从 bank 读取      │
│  │        V' = concat([V, V_ref], dim=1)   │                     │
│  │                                         │                     │
│  │        output = attention(Q, K', V')    │                     │
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

### 核心机制

#### 1. Reference Feature Extraction

```python
def extract_reference_features(
    product_image: Image,
    vae,
    transformer,
    device="cuda"
):
    """
    提取产品图片的 reference features（只做一次）

    Args:
        product_image: 产品图片（PIL Image）
        vae: VAE encoder
        transformer: Wan transformer

    Returns:
        ref_bank: {layer_idx: {"key": K_ref, "value": V_ref}}
    """
    # 1. 编码产品图片到 latent space
    ref_tensor = preprocess_image(product_image)  # [3, H, W]
    with torch.no_grad():
        z_ref = vae.encode([ref_tensor.unsqueeze(1)])[0]  # [16, 1, H', W']

    # 2. 准备 feature extraction
    ref_bank = {}

    def save_ref_features(name):
        def hook(module, input, output):
            # 在 self-attention 中保存 K, V
            hidden_states = input[0]
            K_ref = module.to_k(hidden_states)
            V_ref = module.to_v(hidden_states)
            ref_bank[name] = {
                "key": K_ref.detach().clone(),
                "value": V_ref.detach().clone()
            }
        return hook

    # 3. 注册 hooks 到所有 self-attention 层
    hooks = []
    for name, module in transformer.named_modules():
        if is_self_attention_layer(module):
            hook = module.register_forward_hook(save_ref_features(name))
            hooks.append(hook)

    # 4. Forward pass（只提取 features，不生成）
    with torch.no_grad():
        _ = transformer(
            [z_ref],
            t=torch.zeros(1, device=device) * 0.5,  # 中等噪声水平
            context=dummy_context,
            seq_len=compute_seq_len(z_ref)
        )

    # 5. 移除 hooks
    for hook in hooks:
        hook.remove()

    return ref_bank
```

#### 2. Modified Self-Attention Processor

```python
class ReferenceAttentionProcessor:
    """
    Reference Self-Attention Processor (基于 RefDrop)
    在 self-attention 中注入 reference features
    """
    def __init__(self, ref_bank, ref_strength=1.0):
        """
        Args:
            ref_bank: {layer_name: {"key": K_ref, "value": V_ref}}
            ref_strength: Reference 影响强度（0.0-1.0）
        """
        self.ref_bank = ref_bank
        self.ref_strength = ref_strength

    def __call__(
        self,
        attn_module,
        hidden_states,
        encoder_hidden_states=None,
        layer_name=None,
        **kwargs
    ):
        """
        Modified self-attention with reference injection
        """
        batch_size = hidden_states.shape[0]

        # 1. 计算 Q, K, V（正常流程）
        query = attn_module.to_q(hidden_states)

        if encoder_hidden_states is None:
            # Self-Attention
            key = attn_module.to_k(hidden_states)
            value = attn_module.to_v(hidden_states)
        else:
            # Cross-Attention（不修改）
            key = attn_module.to_k(encoder_hidden_states)
            value = attn_module.to_v(encoder_hidden_states)
            # 直接返回，不注入 reference
            return attn_module.attention(query, key, value)

        # 2. 从 ref_bank 获取 reference features
        if layer_name in self.ref_bank:
            K_ref = self.ref_bank[layer_name]["key"]
            V_ref = self.ref_bank[layer_name]["value"]

            # 3. Expand to match batch size
            K_ref = K_ref.expand(batch_size, -1, -1)
            V_ref = V_ref.expand(batch_size, -1, -1)

            # 4. 拼接 reference features
            # key: [B, seq_len, dim] + [B, ref_seq_len, dim]
            key = torch.cat([key, K_ref * self.ref_strength], dim=1)
            value = torch.cat([value, V_ref * self.ref_strength], dim=1)

        # 5. 计算 attention（Q 不变，K/V 增强）
        output = attn_module.attention(query, key, value)

        return output
```

#### 3. 集成到 FlowAlign

```python
def flowalign_with_reference(
    model,
    vae,
    text_encoder,
    source_video,
    source_prompt,
    target_prompt,
    target_image=None,         # NEW: 产品图片
    ref_strength=1.0,          # NEW: Reference 强度
    **flowalign_params
):
    """
    FlowAlign with Reference Self-Attention
    """
    # Step 1: Extract reference features（如果提供了产品图片）
    if target_image is not None:
        print("Extracting reference features from product image...")
        ref_bank = extract_reference_features(
            product_image=target_image,
            vae=vae,
            transformer=model,
            device=device
        )

        # Step 2: 注册 Reference Attention Processor
        ref_processor = ReferenceAttentionProcessor(
            ref_bank=ref_bank,
            ref_strength=ref_strength
        )
        register_attention_processor(model, ref_processor)
        print(f"Reference attention registered (strength={ref_strength})")
    else:
        ref_bank = None
        print("No product image provided, using vanilla FlowAlign")

    # Step 3: 正常执行 FlowAlign
    # （attention processor 会自动在 forward 时调用）
    output = flowalign(
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        source_video=source_video,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        **flowalign_params
    )

    # Step 4: 清理（移除 processor）
    if ref_bank is not None:
        remove_attention_processor(model)

    return output
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

### 2. Reference Timestep 选择

**问题：** 在哪个噪声水平提取 reference features？

**选项：**

| Timestep | 噪声水平 | 特性 | 适用场景 |
|----------|---------|------|----------|
| t = 0.0 | 无噪声 | 细节丰富，语义精确 | 需要精确外观 |
| t = 0.5 | 中等噪声 | 语义清晰，细节模糊 | 平衡外观和灵活性 |
| t = 1.0 | 纯噪声 | 只有高层语义 | 只控制风格 |

**RefDrop 建议：** 使用多个 timestep 的平均

```python
# 在多个噪声水平提取 features，然后平均
timesteps = [0.0, 0.3, 0.5]
ref_banks = []
for t in timesteps:
    bank = extract_reference_features(image, vae, model, timestep=t)
    ref_banks.append(bank)

# 平均
ref_bank_avg = average_ref_banks(ref_banks)
```

**建议：先用 t=0.5 测试，效果不好再尝试多 timestep**

### 3. Reference Strength 控制

**作用：** 控制产品图片的影响强度

```python
# 在拼接时乘以 strength
key = torch.cat([key, K_ref * ref_strength], dim=1)
value = torch.cat([value, V_ref * ref_strength], dim=1)
```

**参数范围：**
- `ref_strength = 0.0`: 无 reference（等价于原始 FlowAlign）
- `ref_strength = 0.5`: 中等影响
- `ref_strength = 1.0`: 完全 reference
- `ref_strength > 1.0`: 增强 reference（可能过拟合）

**建议：** 从 1.0 开始，根据效果调整

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

2. **Strength 消融**
   - ref_strength = 0.0, 0.3, 0.5, 0.7, 1.0, 1.5

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
       target_image: jewelry/NECK001.jpg  # NEW

   flowalign:
       ref_strength: 1.0  # NEW
       ref_timestep: 0.5  # NEW
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

**原因：** `ref_strength` 太大，或者所有层都注入

**解决：**
- 降低 `ref_strength`（试试 0.3-0.7）
- 只在 deep layers 注入（控制细节，不影响运动）

### 问题 2: Reference 太弱，外观不像

**表现：** 还是主要靠 text prompt，产品图片影响小

**原因：** `ref_strength` 太小，或者只在少数层注入

**解决：**
- 增大 `ref_strength`（试试 1.0-1.5）
- 在更多层注入（middle + deep）

### 问题 3: 时序不一致（闪烁）

**表现：** 相邻帧之间外观跳变

**原因：** Reference features 是静态的，每帧独立参考

**解决：**
1. **降低 ref_strength**（减少干扰）
2. **只在特定层注入**（保留其他层的 temporal attention）
3. **增加 temporal smoothing**
   ```python
   # 在相邻帧之间插值 reference strength
   ref_strength_per_frame = smoothly_vary(base_strength, num_frames)
   ```

### 问题 4: 计算成本高

**表现：** 推理变慢

**原因：** K, V 的序列长度增加（拼接了 reference）

**解决：**
- 只在部分层注入（减少计算）
- 压缩 reference features（降低维度）
- 使用 Flash Attention（优化 attention 计算）

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

5. **FlowAlign (arXiv 2024)**
   [Training-Free Consistent Text-to-Image Generation](https://arxiv.org/abs/2405.23145)
   - 我们的 baseline 方法

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
       --config config/pvtt/test01_watch_to_bracelet.yaml \
       --target_image ../../data/pvtt-benchmark/images/jewelry/bracelet.jpg \
       --ref_strength 1.0
   ```

4. **对比结果：**
   - Baseline: text only
   - + Reference: text + image

### 后续优化（Phase 2-3）

- 参数调优
- 批量实验
- 写入论文

---

**Status:** Design Complete
**Next:** Implementation
**ETA:** Phase 1 可在 1-2 天内完成
