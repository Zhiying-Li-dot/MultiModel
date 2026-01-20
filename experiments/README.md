# PVTT 实验记录

本目录统一管理所有 PVTT 项目的实验结果、配置、分析报告等。

## 目录结构

```
experiments/
├── README.md                    # 本文件（实验索引）
├── logs/                        # 实验日志（格式：主题_日期.md）
│   ├── flowedit-baseline_2026-01-05.md
│   ├── refdrop_2026-01-09.md
│   ├── flux-ti2v-compositional_2026-01-10.md
│   ├── ti2v-flowedit_2026-01-12.md
│   ├── rfsolver-random-noise_2026-01-19.md
│   ├── rfsolver-ti2v-inversion_2026-01-20.md
│   ├── rfsolver-shift-ablation_2026-01-20.md
│   └── rfsolver-frames-ablation_2026-01-20.md
├── results/                     # 实验结果（视频、图片等）
│   ├── flowalign-wan2.1/       # FlowAlign (Wan2.1) 基线结果
│   ├── flowedit-wan2.1/        # FlowEdit (Wan2.1) 基线结果
│   └── compositional/          # 组合方法结果
└── analysis/                    # 实验分析、对比报告
```

---

## 实验日志索引

### 2026-01

| 日期 | 主题 | 文件 | 关键结论 |
|------|------|------|---------|
| 01-20 | RF-Solver 帧数消融 | [rfsolver-frames-ablation_2026-01-20.md](logs/rfsolver-frames-ablation_2026-01-20.md) | **std≠质量**；帧数多反而后续帧退化；推荐 17-25 帧 |
| 01-20 | RF-Solver Shift 消融 | [rfsolver-shift-ablation_2026-01-20.md](logs/rfsolver-shift-ablation_2026-01-20.md) | **shift=0.5 最佳**，std=1.06；shift 越小越精确 |
| 01-20 | RF-Solver TI2V Inversion | [rfsolver-ti2v-inversion_2026-01-20.md](logs/rfsolver-ti2v-inversion_2026-01-20.md) | 中点不应加 TI2V 条件；RF-Solver v2 std=0.88 |
| 01-19 | RF-Solver vs 随机噪声 | [rfsolver-random-noise_2026-01-19.md](logs/rfsolver-random-noise_2026-01-19.md) | **随机噪声效果最好**，Inversion 反而有害 |
| 01-12 | TI2V FlowEdit | [ti2v-flowedit_2026-01-12.md](logs/ti2v-flowedit_2026-01-12.md) | 失败：首帧条件破坏 FlowEdit velocity 假设 |
| 01-10 | Flux.2 + TI2V 组合方法 | [flux-ti2v-compositional_2026-01-10.md](logs/flux-ti2v-compositional_2026-01-10.md) | **当前最佳方案**；产品替换效果最好 |
| 01-09 | RefDrop 图像引导 | [refdrop_2026-01-09.md](logs/refdrop_2026-01-09.md) | c=0.05 可用；Prompt 主导，参考图只微调 |
| 01-05 | FlowEdit/FlowAlign 基线 | [flowedit-baseline_2026-01-05.md](logs/flowedit-baseline_2026-01-05.md) | FlowAlign (Wan2.1) 确定为基线 |

---

## 当前最佳方案

**Flux.2 + TI2V 组合方法**（随机噪声）：

```
目标首帧 (Flux.2 编辑) + 随机噪声 ──▶ TI2V 去噪 ──▶ 目标视频
```

详见 [flux-ti2v-compositional_2026-01-10.md](logs/flux-ti2v-compositional_2026-01-10.md) 和 [rfsolver-random-noise_2026-01-19.md](logs/rfsolver-random-noise_2026-01-19.md)。

---

## 方法演进总结

```
FlowEdit/FlowAlign 基线 (01-05)
         │
         ▼
RefDrop 图像引导 (01-09) ──▶ 效果有限，c 值必须极小
         │
         ▼
Flux.2 + TI2V 组合方法 (01-10) ──▶ 当前最佳！产品替换效果好
         │
         ▼
TI2V FlowEdit (01-12) ──▶ 失败，首帧条件与 FlowEdit 不兼容
         │
         ▼
RF-Solver Inversion (01-19) ──▶ 随机噪声更优，Inversion 无益
         │
         ▼
RF-Solver v2 (01-20) ──▶ 中点不加条件，std=0.88，适合保留运动
```

---

## 命名规范

### 实验日志

格式：`{主题}_{日期}.md`

示例：
- `flowedit-baseline_2026-01-05.md`
- `rfsolver-ti2v-inversion_2026-01-20.md`

### 结果视频

格式：`{test_id}_{method}_{source}_to_{target}.mp4`

示例：
- `test01_flowalign_watch_to_bracelet.mp4`
- `ti2v_rfsolver_v2.mp4`

---

## 添加新实验

1. 在 `experiments/logs/` 创建新日志文件：`{主题}_{日期}.md`
2. 更新本文件的实验索引表
3. 结果文件保存到 `experiments/results/{method}/`
4. 提交到 git

---

## 下一步计划

- [ ] 在更多案例上验证随机噪声方法
- [ ] 改进 Flux.2 首帧生成质量
- [ ] 改进 TI2V 运动控制（motion prompt、运动轨迹条件）
- [ ] 计算定量指标（CLIP-score, FVD 等）
