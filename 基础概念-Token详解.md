# Token详解：AI模型的基本单位

Token是AI模型处理信息的基本单位。在不同的场景下，Token有不同的含义。

## 1. 文本Token（最常见）

### 1.1 什么是文本Token？

**Token = 文本的最小处理单位**

就像句子可以分解成词语，文本也可以分解成Token。

```
示例1：英文分词
原始文本: "Hello, world!"
Tokens:   ["Hello", ",", "world", "!"]
Token数量: 4个

示例2：中文分词
原始文本: "你好世界"
Tokens可能是:
方法1（字符级）: ["你", "好", "世", "界"]  # 4个token
方法2（词级）:   ["你好", "世界"]          # 2个token
方法3（子词级）: ["你", "好", "世界"]      # 3个token
```

### 1.2 为什么需要Token？

```
计算机不能直接理解文字
    ↓
需要把文字转换成数字
    ↓
首先把文字切分成Token
    ↓
再把每个Token转换成数字（Token ID）
    ↓
最后转换成向量（Embedding）
    ↓
模型才能处理
```

### 1.3 完整流程演示

```python
# 步骤1: 原始文本
text = "I love AI!"

# 步骤2: 分词（Tokenization）
tokens = ["I", "love", "AI", "!"]

# 步骤3: 转换为Token ID
# 每个token在词表中有一个唯一的ID
token_ids = [100, 1234, 5678, 999]
#            "I"  "love" "AI"  "!"

# 步骤4: 转换为向量（Embedding）
# 每个ID对应一个向量
embeddings = [
    [0.1, 0.2, 0.3, ...],  # "I"的768维向量
    [0.5, 0.6, 0.1, ...],  # "love"的768维向量
    [0.9, 0.2, 0.8, ...],  # "AI"的768维向量
    [0.3, 0.1, 0.4, ...],  # "!"的768维向量
]

# 步骤5: 模型处理
# Transformer接收这些向量进行计算
```

### 1.4 实际代码示例

```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 原始文本
text = "Hello, how are you?"

# 分词
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# 输出: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
# 注意: Ġ表示这个token前面有空格

# 转换为ID
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
# 输出: [15496, 11, 703, 389, 345, 30]

# 解码回文本
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
# 输出: "Hello, how are you?"
```

---

## 2. 不同的Tokenization方法

### 2.1 方法对比

```
┌──────────────────────────────────────────────────────────┐
│              三种主要的Tokenization方法                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. 字符级 (Character-level)                            │
│     文本: "Hello"                                        │
│     Token: ['H', 'e', 'l', 'l', 'o']                    │
│     ✓ 词表小（只需26个字母+符号）                        │
│     ✗ Token序列太长                                      │
│     ✗ 难以学习语义                                       │
│                                                          │
│  2. 词级 (Word-level)                                   │
│     文本: "Hello world"                                  │
│     Token: ['Hello', 'world']                           │
│     ✓ 符合直觉                                          │
│     ✗ 词表巨大（需要几十万个词）                         │
│     ✗ 无法处理未登录词（OOV）                            │
│                                                          │
│  3. 子词级 (Subword-level) ⭐ 最常用                    │
│     文本: "unbelievable"                                │
│     Token: ['un', 'believ', 'able']                     │
│     ✓ 词表大小适中（3万-5万）                            │
│     ✓ 可以处理任何新词                                   │
│     ✓ 平衡了字符级和词级的优缺点                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 子词分词算法

#### BPE (Byte Pair Encoding) - GPT使用

```python
# BPE的核心思想：从字符开始，逐步合并高频出现的字符对

# 初始词表: 所有字符
vocab = ['a', 'b', 'c', ..., 'z']

# 统计字符对频率
text = "low low low lower"
# 字符级: ['l','o','w', ' ', 'l','o','w', ...]

# 找到最高频的字符对: 'l'+'o' 出现很多次
# 合并它们，创建新token 'lo'
vocab.append('lo')

# 继续迭代...
# 'lo'+'w' → 'low'
# 'low'+'er' → 'lower'

# 最终词表包含: 字符 + 常见子词 + 完整单词
```

#### WordPiece - BERT使用

```python
# WordPiece类似BPE，但使用不同的合并策略

text = "playing"
tokens = ['play', '##ing']  # ##表示这不是词的开头

text = "unwalkable"
tokens = ['un', '##walk', '##able']
```

#### SentencePiece - LLaMA使用

```
特点：
- 不依赖于空格分词
- 直接从原始文本学习
- 支持任何语言（包括中文、日文等无空格语言）
```

### 2.3 中英文对比

```python
# 英文
text_en = "I love programming"
tokens_en = ['I', 'love', 'program', 'ming']  # 4个token

# 中文（GPT-2分词器）
text_cn = "我爱编程"
tokens_cn = ['我', '爱', '编', '程']  # 4个token
# 注意：中文通常每个字是一个token，效率较低

# 中文（专门的中文分词器）
text_cn = "我爱编程"
tokens_cn = ['我', '爱', '编程']  # 3个token
# 更好的中文分词器会识别词语

# 实际token数对比
# 英文: 100个单词 ≈ 75-100个token
# 中文: 100个字 ≈ 150-200个token（用英文分词器）
# 中文: 100个字 ≈ 80-120个token（用中文分词器）
```

---

## 3. 视觉Token

### 3.1 图像如何变成Token？

```
图像 (224×224×3)
    ↓
分成小块 (Patches)
    ↓
每个小块变成一个Token
    ↓
Vision Transformer处理

示例：ViT-L/14
─────────────────────────────────
输入图像: 224×224 pixels
Patch大小: 14×14 pixels
Patch数量: (224/14) × (224/14) = 16×16 = 256个
每个Patch = 一个Token
加上1个 [CLS] token
总共: 257个视觉Token
```

### 3.2 可视化演示

```
原始图像 (224×224)              Patch分割 (14×14每块)
┌────────────────────┐         ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│                    │         │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
│                    │         ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│      一只猫        │  ──►   │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
│                    │         ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│                    │         │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
└────────────────────┘         └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
                                    16×16 = 256个patches

每个Patch:
┌────────┐
│14×14   │  ──► 展平 ──► 投影 ──► 一个Token向量
│pixels  │      196维      768维    (或1024维)
└────────┘
```

### 3.3 代码实现

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """将图像分割成patches并转换为token"""

    def __init__(self, img_size=224, patch_size=14, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 256

        # 用卷积实现patch分割和投影
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, 3, 224, 224) 输入图像
        return: (B, 256, 768) 视觉token
        """
        # 卷积提取patches
        x = self.proj(x)  # (B, 768, 16, 16)

        # 展平空间维度
        x = x.flatten(2)  # (B, 768, 256)
        x = x.transpose(1, 2)  # (B, 256, 768)

        return x

# 使用示例
patch_embed = PatchEmbedding()
image = torch.randn(1, 3, 224, 224)  # 一张图像
tokens = patch_embed(image)
print(f"视觉Token形状: {tokens.shape}")  # (1, 256, 768)
print(f"Token数量: {tokens.shape[1]}")   # 256
```

---

## 4. 多模态中的Token

### 4.1 多模态模型如何处理Token？

```
多模态模型需要同时处理图像Token和文本Token

示例：LLaVA处理 "描述这张图片"
──────────────────────────────────────────────────

输入：
- 图像: cat.jpg
- 文本: "描述这张图片"

步骤1: 图像编码
cat.jpg ──► ViT ──► 256个视觉token (每个768维)
            ↓
        投影层
            ↓
        256个视觉token (每个4096维，匹配LLM)

步骤2: 文本编码
"描述这张图片" ──► Tokenizer ──► [描述, 这, 张, 图片]
                              ↓
                        [ID1, ID2, ID3, ID4]
                              ↓
                        4个文本token (每个4096维)

步骤3: 拼接Token序列
[BOS] [IMG_START] [256个视觉token] [IMG_END] [4个文本token]

总共: 1 + 1 + 256 + 1 + 4 = 263个token

步骤4: LLM处理
所有263个token一起输入到LLaMA
LLM用Self-Attention让视觉token和文本token交互
输出: "这张图片展示了一只橙色的猫..."
```

### 4.2 Token序列可视化

```
LLaVA的Token序列：
┌─────────────────────────────────────────────────────────────┐
│ [BOS]  <IMG>  [V1][V2]...[V256]  </IMG>  [T1][T2][T3][T4]  │
│   │      │           │              │           │           │
│ 开始   图像    256个视觉token    图像结束  文本token         │
│ token  标记                       标记                      │
└─────────────────────────────────────────────────────────────┘
        ↓                    ↓                    ↓
     1个token          258个token            4个token

                总共: 263个token

BLIP-2的Token序列（使用Q-Former压缩）：
┌─────────────────────────────────────────────────────────────┐
│ [BOS]  [Q1][Q2]...[Q32]  [T1][T2][T3][T4]                   │
│   │           │                  │                          │
│ 开始    32个压缩的视觉token    文本token                     │
└─────────────────────────────────────────────────────────────┘
        ↓            ↓              ↓
     1个token    32个token      4个token

                总共: 37个token

对比：
- LLaVA: 更多token，保留更多细节，计算量大
- BLIP-2: 更少token，压缩信息，计算高效
```

---

## 5. Token的重要性

### 5.1 为什么Token数量很重要？

```
Token数量直接影响：

1. 计算成本
   Self-Attention的复杂度 = O(N²)
   N = token数量

   示例：
   100个token: 100² = 10,000次计算
   1000个token: 1000² = 1,000,000次计算 (100倍!)

2. 内存占用
   每个token需要存储：
   - Token embedding: 4096维 × 4字节 = 16KB
   - Attention缓存: 更多内存

   1000个token ≈ 16MB+

3. 上下文长度限制
   GPT-3.5: 4096 token限制
   GPT-4: 8192 token限制
   Claude: 100K token限制

   一张图可能就占用200-600个token！

4. API成本
   OpenAI API按token计费
   输入: $0.03 / 1K tokens
   输出: $0.06 / 1K tokens
```

### 5.2 优化Token使用

```python
# 问题：图像占用太多token
image_tokens = 256  # LLaVA
text_tokens = 50
total = 306  # 太多了！

# 解决方案1: 使用Q-Former压缩
image_tokens = 32   # BLIP-2的Q-Former
text_tokens = 50
total = 82  # 减少了73%！

# 解决方案2: 使用更小的patch size
# ViT-L/14: 256 tokens (14×14 patch)
# ViT-L/16: 196 tokens (16×16 patch)

# 解决方案3: 只用CLS token
image_tokens = 1    # 只用全局表示
text_tokens = 50
total = 51  # 最少，但可能损失细节
```

---

## 6. 实际应用示例

### 6.1 计算文本的Token数

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 示例1: 短文本
text1 = "Hello world"
tokens1 = tokenizer.encode(text1)
print(f"'{text1}' 有 {len(tokens1)} 个token")
# 输出: 'Hello world' 有 2 个token

# 示例2: 长文本
text2 = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。
它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
"""
tokens2 = tokenizer.encode(text2)
print(f"长文本有 {len(tokens2)} 个token")
# 输出: 长文本有 100+ 个token

# 示例3: 代码
code = """
def hello_world():
    print("Hello, World!")
    return True
"""
tokens_code = tokenizer.encode(code)
print(f"代码有 {len(tokens_code)} 个token")
# 代码通常token数更多，因为有特殊符号

# 经验法则：
# 英文: 100个词 ≈ 75-100个token
# 中文: 100个字 ≈ 150-250个token (用英文tokenizer)
# 代码: 100行 ≈ 200-500个token
```

### 6.2 多模态Token计算

```python
# 计算一个多模态输入的总token数

def count_multimodal_tokens(image_path, text, model_type="llava"):
    """
    计算多模态输入的token总数
    """
    # 文本token
    text_tokens = len(tokenizer.encode(text))

    # 图像token
    if model_type == "llava":
        image_tokens = 256  # ViT-L/14的输出
    elif model_type == "blip2":
        image_tokens = 32   # Q-Former的输出
    elif model_type == "clip":
        image_tokens = 1    # 只用CLS token

    # 特殊token
    special_tokens = 3  # [BOS], <IMG>, </IMG>

    total = special_tokens + image_tokens + text_tokens

    return {
        'image_tokens': image_tokens,
        'text_tokens': text_tokens,
        'special_tokens': special_tokens,
        'total': total
    }

# 示例
result = count_multimodal_tokens(
    "cat.jpg",
    "请详细描述这张图片中的内容，包括物体、颜色、位置等信息。",
    model_type="llava"
)

print(f"图像token: {result['image_tokens']}")
print(f"文本token: {result['text_tokens']}")
print(f"总token数: {result['total']}")

# 输出:
# 图像token: 256
# 文本token: 35
# 总token数: 294
```

---

## 7. 常见问题

### Q1: 为什么中文比英文消耗更多token？

```
原因：大多数tokenizer是为英文设计的

英文: "I love AI"
Tokens: ['I', 'love', 'AI']  # 3个token

中文: "我爱AI"
Tokens: ['我', '爱', 'AI']   # 3个token（好的情况）
      或
Tokens: ['â', '我', 'ç', '±', 'ªAI']  # 更多token（差的情况）

解决方案：
- 使用针对中文优化的tokenizer（如ChatGLM的tokenizer）
- 使用多语言tokenizer（如XLM-RoBERTa）
```

### Q2: Token数量和模型大小有关系吗？

```
有关系，但不是直接关系：

Token数量 → 影响单次推理的计算量
模型参数量 → 影响模型总大小和能力

示例：
- GPT-3 (175B参数) 可以处理 2048 tokens
- GPT-4 (参数未公开) 可以处理 8192 tokens
- Claude-2 (参数未公开) 可以处理 100K tokens

更大的模型通常支持更长的上下文，但不是必然的。
```

### Q3: 如何减少Token使用？

```
方法1: 压缩文本
- 去掉不必要的词
- 使用简洁表达

方法2: 使用摘要
- 先总结长文档
- 再输入摘要

方法3: 分块处理
- 将长文本分成多个块
- 分别处理后合并结果

方法4: 选择合适的模型
- 需要细节：用LLaVA (256 tokens)
- 需要效率：用BLIP-2 (32 tokens)
```

---

## 总结

### Token的本质
```
Token = 模型处理信息的基本单位

就像：
- 文章由句子组成
- 句子由词语组成
- AI模型的输入由Token组成

理解Token是理解AI模型的第一步！
```

### 关键要点

1. **文本Token**: 通过分词器将文本分割成子词
2. **视觉Token**: 将图像分割成patches，每个patch是一个token
3. **Token数量**: 影响计算成本、内存、上下文长度
4. **多模态**: 需要拼接视觉token和文本token
5. **优化**: Q-Former等技术可以压缩token数量

### 实用工具

```python
# 计算token数
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
token_count = len(tokenizer.encode("your text"))

# 查看实际token
tokens = tokenizer.tokenize("your text")
print(tokens)

# 在线工具
# https://platform.openai.com/tokenizer
# 可以可视化看到文本如何被分成token
```
