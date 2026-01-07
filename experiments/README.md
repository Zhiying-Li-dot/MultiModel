# PVTT 实验记录

本目录统一管理所有 PVTT 项目的实验结果、配置、分析报告等。

## 目录结构

```
experiments/
├── README.md                    # 本文件
├── results/                     # 实验结果（视频、图片等）
│   ├── flowalign-wan2.1/       # FlowAlign (Wan2.1) 基线结果
│   │   ├── test01_flowalign_watch_to_bracelet.mp4
│   │   ├── test02_flowalign_tray_to_flowers.mp4
│   │   └── ...
│   ├── flowedit-wan2.1/        # FlowEdit (Wan2.1) 基线结果
│   └── our-method/             # 我们的方法实验结果
├── configs/                     # 实验配置（自动生成）
│   └── flowalign-wan2.1/
│       ├── test01.yaml
│       └── ...
├── logs/                        # 实验日志
│   └── flowalign-wan2.1/
│       ├── test01.log
│       └── ...
└── analysis/                    # 实验分析、对比报告
    ├── baseline-comparison.md
    └── metrics/
        └── flowalign-results.csv
```

## 组织原则

1. **按方法分类** - 每个方法（baseline或我们的方法）有独立子目录
2. **统一命名** - 实验结果文件名包含方法名和转换任务
3. **版本管理** - 重要结果提交到 git，大文件用 git-lfs
4. **可追溯** - 保留配置文件和日志，方便复现

## 命名规范

### 结果视频
格式：`{test_id}_{method}_{source}_to_{target}.mp4`

示例：
- `test01_flowalign_watch_to_bracelet.mp4`
- `test02_our_method_tray_to_flowers.mp4`

### 实验日志
格式：`{test_id}_{method}_{timestamp}.log`

示例：
- `test01_flowalign_20260107_161800.log`

## 当前实验

### FlowAlign-Wan2.1 基线

| 测试ID | 类别 | 转换 | 状态 | 结果文件 |
|--------|------|------|------|---------|
| test01 | Jewelry | Watch → Bracelet | ✅ | test01_flowalign_watch_to_bracelet.mp4 |
| test02 | Home | Tray → Flowers | ✅ | test02_flowalign_tray_to_flowers.mp4 |
| test03 | Toys | Stacker → Toy | ✅ | test03_flowalign_stacker_to_ridetoy.mp4 |
| test04 | Clothing | Socks → Skirt | ✅ | test04_flowalign_socks_to_skirt.mp4 |

**参数设置：**
- Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Frames: 49 (固定取前49帧)
- Resolution: 480p (832x480 或 480x832)
- Steps: 50
- Strength: 0.7
- Guidance Scale: 19.5
- Attention Masking: Layers 11-17

## 添加新实验

1. 在 `baseline/flowedit-wan2.1/run_experiments.py` 中添加实验配置
2. 运行实验：`python run_experiments.py --experiments new_test`
3. 结果自动保存到 `experiments/results/{method}/`
4. 更新本文档的实验记录表

## 实验分析

### 下一步计划
- [ ] 完成更多样本对的测试
- [ ] 对比 FlowAlign vs FlowEdit
- [ ] 计算定量指标（CLIP-score, FVD等）
- [ ] 用户研究评估
- [ ] 实现我们的改进方法
