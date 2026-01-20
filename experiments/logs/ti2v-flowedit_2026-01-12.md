# TI2V FlowEdit 实验

**日期**: 2026-01-10 ~ 2026-01-12
**主题**: FlowEdit 与 Wan2.2 TI2V-5B 结合
**测试用例**: bracelet → pearl necklace
**结果**: 失败

---

## 实验背景

尝试将 FlowEdit 算法与 Wan2.2 TI2V-5B 模型结合，利用首帧条件引导视频编辑。

### FlowEdit 2-Branch 结构

```
Source: Vt_src = CFG(V(Zt_src, source_prompt, source_first_frame))
Target: Vt_tar = CFG(V(Zt_tar, target_prompt, target_first_frame))
Update: Zt_edit += dt * (Vt_tar - Vt_src)
```

### 与 FlowAlign 3-Branch 的区别

| 方法 | 分支数 | 特点 |
|------|--------|------|
| FlowEdit | 2 | source/target 独立 CFG |
| FlowAlign | 3 | vq_source, vp_source, vp_target + zeta_scale 正则化 |

---

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | Wan2.2-TI2V-5B |
| 源视频 | bracelet_shot1.mp4 (1280x1024, 49帧) |
| 目标首帧 | target_frame1.png (Flux.2 生成) |
| 输出分辨率 | 1056x832 (best_output_size) |
| Steps | 50 |
| Strength | 0.7 |
| Source/Target CFG | 5.0 |

---

## 实验结果

| 版本 | 分辨率 | 结果 |
|------|--------|------|
| v1 (480p) | 832x480 | 模糊 |
| v2 (highres) | 1056x832 | 错误 |

**参考对比**: wan22_flowalign.mp4 (FlowAlign 3-branch) 效果正确

### 结果截图 (v2 highres - 失败)

| Frame 0 | Frame 9 |
|---------|---------|
| ![Frame 0](../results/compositional/debug_highres_frame0.png) | ![Frame 9](../results/compositional/debug_highres_frame9.png) |

**FlowAlign T2V vs TI2V FlowEdit 对比** (Frame 24):

| FlowAlign T2V (正确) | TI2V FlowEdit (错误) |
|---------------------|---------------------|
| ![FlowAlign](../results/debug_flowalign_t2v_f24.png) | ![FlowEdit](../results/debug_ti2v_flowedit_f24.png) |

---

## 问题分析

### 1. FlowEdit vs FlowAlign 算法差异

```python
# FlowEdit: 直接用速度差更新
Zt_edit += dt * (Vt_tar - Vt_src)

# FlowAlign: 有额外正则化
Zt_edit += dt * (Vt_tar - Vt_src) + zeta_scale * consistency_term
```

TI2V 模式可能需要 FlowAlign 的正则化项。

### 2. 首帧条件处理

- Source 分支用 source_first_frame 条件
- Target 分支用 target_first_frame 条件
- 两个不同的首帧条件可能导致不一致

### 3. CFG 设置

当前 source_cfg = target_cfg = 5.0，可能需要不同的 CFG 配比。

---

## 核心发现

### FlowEdit 与 TI2V 不兼容

1. **FlowEdit 假设 source/target 共享相同的结构**
   - 在 T2V 模式下，source/target 只有 prompt 不同
   - 在 TI2V 模式下，source/target 有不同的首帧条件

2. **首帧条件破坏了 FlowEdit 的前提**
   - FlowEdit 的 velocity 差值假设两个分支在相同结构上
   - 不同首帧导致两个分支的 latent 结构不同

3. **FlowAlign 的正则化是必要的**
   - FlowAlign 的 zeta_scale 项帮助对齐两个分支
   - 没有正则化，TI2V 模式下 FlowEdit 失败

---

## 结论

1. **TI2V + FlowEdit 组合失败**
2. **原因**: 首帧条件破坏 FlowEdit 的 velocity 差值假设
3. **替代方案**: 使用 FlowAlign (3-branch) 或放弃端到端编辑

---

## 后续方向

- [x] 分析 FlowAlign 为什么有效 → zeta_scale 正则化
- [ ] 尝试在 FlowEdit 中加入 zeta_scale 正则化
- [ ] 尝试不同的首帧条件组合
- [ ] 对比 T2V 模式下的 FlowEdit（无首帧条件）

---

## 相关文件

- 实现: `baseline/compositional-flux-ti2v/scripts/ti2v_flowedit.py`
- 参考: `baseline/flowedit-wan2.2/flowalign_ti2v.py`
- 结果: `experiments/results/compositional/ti2v_flowedit_*.mp4`
