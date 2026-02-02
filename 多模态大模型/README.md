# 多模态大模型学习笔记

本仓库包含多模态大模型（Multimodal Large Language Models）的核心知识点详解。

## 目录

1. [基础概念](./01-基础概念.md) - 模态定义、多模态融合、对齐与映射
2. [核心架构](./02-核心架构.md) - 视觉编码器、语言模型、连接模块、典型架构
3. [训练范式](./03-训练范式.md) - 预训练、指令微调、RLHF
4. [关键技术](./04-关键技术.md) - 视觉Token化、位置编码、注意力机制、高分辨率处理
5. [数据工程](./05-数据工程.md) - 预训练数据、指令数据构建、数据质量过滤
6. [评估体系](./06-评估体系.md) - 基准测试、能力维度、评估方法
7. [前沿方向](./07-前沿方向.md) - 视频理解、多模态生成、多模态Agent
8. [架构深度解析与面试指南](./08-架构深度解析与面试指南.md) - LLaVA/BLIP-2架构详解、连接模块对比、面试题库

## 学习路线建议

1. 先理解单模态模型（ViT、LLM）
2. 从LLaVA入手，理解最简单的架构
3. 阅读经典论文：CLIP、BLIP-2、LLaVA
4. 动手实践：用开源代码训练/微调

## 参考论文

- [CLIP](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [ViT](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words
- [BLIP-2](https://arxiv.org/abs/2301.12597) - Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- [LLaVA](https://arxiv.org/abs/2304.08485) - Visual Instruction Tuning
- [Flamingo](https://arxiv.org/abs/2204.14198) - A Visual Language Model for Few-Shot Learning

## License

MIT License
