# Baseline 设计

## 选择 Training-Free Video Editing 作为 Baseline

**理由**：
1. PVTT 本质是"保持模板结构/运镜，替换产品"—— 与 video editing 目标一致
2. 可以快速验证任务可行性，展示现有方法的不足
3. 论文需要 baseline 对比
4. 不需要训练，快速出结果

---

## FlowEdit：ICCV 2025 Best Student Paper

| 项目 | 信息 |
|------|------|
| **论文** | FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models |
| **会议** | ICCV 2025 **Best Student Paper** |
| **特点** | Inversion-free, optimization-free, model agnostic |
| **GitHub** | https://github.com/fallenshock/FlowEdit |

**核心创新**：
- 构造 ODE 直接映射源分布到目标分布
- 不需要 inversion，比传统方法 transport cost 更低
- 支持 SD3, FLUX, Hunyuan, LTX-Video

---

## WAN2.1 + FlowEdit 实现

| 项目 | 链接 | 说明 |
|------|------|------|
| **Awesome-Training-Free-WAN2.1-Editing** | https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing | 推荐使用 |
| **ComfyUI-MagicWan** | https://github.com/zackabrams/ComfyUI-MagicWan | ComfyUI 实现 |

**WANAlign2.1 特点**：
- Inversion-free video editing framework
- 基于 FlowEdit/FlowAlign
- 引入 DIFS (Decoupled Inversion-Free Sampling) 控制编辑区域
- WAN2.1 使用 DiT 架构，可直接采用 FlowEdit

---

## 作为 Baseline 的优势

1. ✅ **SOTA**：ICCV 2025 Best Student Paper
2. ✅ **基于 WAN**：与研究方向一致
3. ✅ **Training-Free**：快速验证
4. ✅ **开源可用**：有现成代码
5. ✅ **技术先进**：Inversion-free 是当前趋势

---

## 预期问题（作为 PVTT motivation）

基于 Training-Free 方法的局限性，预期会有以下问题：

1. **主体时序一致性差**：产品在帧间闪烁或形变
2. **不完全跟随指令编辑**：可能只改变部分区域或风格
3. **产品细节丢失**：无法保持产品的精确外观
4. **形态迁移困难**：不同形态产品（手链→项链）的替换效果差

这些问题将成为 PVTT 方法需要解决的核心挑战。
