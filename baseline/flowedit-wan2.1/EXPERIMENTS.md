# PVTT 实验管理

统一的实验管理脚本，自动处理配置生成、GPU选择、批量运行等问题。

## 为什么需要这个脚本？

之前运行实验遇到的问题：
1. **路径错误** - 手动配置YAML容易写错相对路径
2. **Token匹配失败** - 多词blend需要手动调整为单词
3. **GPU管理混乱** - 需要手动检查GPU占用、指定CUDA_VISIBLE_DEVICES
4. **重复工作** - 每个实验都要单独配置和运行

现在所有问题一键解决！

## 快速开始

### 1. 运行所有实验（并行）

```bash
cd ~/pvtt/baseline/flowedit-wan2.1
python run_experiments.py
```

自动：
- 查找可用GPU
- 生成配置文件
- 并行启动所有实验
- 显示进度和结果

### 2. 运行指定实验

```bash
# 只运行test01和test02
python run_experiments.py --experiments test01_watch_to_bracelet test02_tray_to_flowers
```

### 3. 顺序运行（节省显存）

```bash
# 一个接一个运行，每次只占用一个GPU
python run_experiments.py --sequential
```

### 4. 指定GPU

```bash
# 在GPU 3上顺序运行
python run_experiments.py --sequential --gpu 3
```

## 实验配置

所有实验参数在 `run_experiments.py` 中统一管理：

```python
EXPERIMENTS = {
    "test01_watch_to_bracelet": {
        "video_path": "jewelry/JEWE001.mp4",
        "source_prompt": "...",
        "target_prompt": "...",
        "source_blend": "watch",  # 单词，避免Token匹配失败
        "target_blend": "bracelet",
    },
    # ...
}
```

修改参数只需改一处，所有实验自动同步。

## 默认参数

### FlowAlign参数
```python
DEFAULT_FLOWALIGN_PARAMS = {
    "strength": 0.7,
    "target_guidance_scale": 19.5,
    "flag_attnmask": True,
    "zeta_scale": 1e-3,
    "bg_zeta_scale": 1e-3,
}
```

### 推理参数
```python
DEFAULT_INFERENCE_PARAMS = {
    "num_inference_step": 50,
}
```

## 目录结构

脚本自动处理所有路径：

```
pvtt/
├── baseline/flowedit-wan2.1/   # 基线方法代码
│   ├── run_experiments.py      # 实验管理脚本
│   ├── awesome_wan_editing.py  # 主程序
│   └── config/pvtt/            # 自动生成的配置文件
│       ├── test01_watch_to_bracelet.yaml
│       ├── test02_tray_to_flowers.yaml
│       └── ...
├── experiments/                # 实验结果（统一管理）
│   └── results/
│       └── flowalign-wan2.1/   # FlowAlign 实验结果
│           ├── test01_flowalign_watch_to_bracelet.mp4
│           ├── test02_flowalign_tray_to_flowers.mp4
│           └── ...
└── data/pvtt-benchmark/videos/ # 输入视频
    ├── jewelry/JEWE001.mp4
    ├── home/HOME002.mp4
    └── ...
```

**目录说明：**
- `baseline/` - 只包含基线方法的代码
- `experiments/` - 统一存放所有实验结果、记录、分析
- `data/` - 原始数据集

## 添加新实验

只需在 `EXPERIMENTS` 字典中添加：

```python
"test05_new_experiment": {
    "video_path": "category/VIDEO.mp4",
    "source_prompt": "Description of source...",
    "target_prompt": "Description of target...",
    "source_blend": "source",  # 单词！
    "target_blend": "target",  # 单词！
},
```

然后运行：
```bash
python run_experiments.py --experiments test05_new_experiment
```

## GPU自动选择

脚本会自动：
1. 检测所有GPU的内存使用
2. 选择内存使用<10GB的GPU
3. 并行模式下为每个实验分配不同GPU
4. 避免OOM错误

## 输出说明

成功运行后会看到：

```
==============================================================
Experiment Results Summary
==============================================================
✓ SUCCESS    test01_watch_to_bracelet
✓ SUCCESS    test02_tray_to_flowers
✓ SUCCESS    test03_stacker_to_ridetoy
✓ SUCCESS    test04_socks_to_skirt
```

结果视频在项目根目录的 `experiments/results/flowalign-wan2.1/` 目录。

## 远程运行

在5090机器上：

```bash
ssh 5090
cd ~/pvtt/baseline/flowedit-wan2.1
export HF_ENDPOINT=https://hf-mirror.com  # 可选，脚本会自动设置

# 并行运行所有实验
~/.conda/envs/wan/bin/python run_experiments.py

# 或顺序运行
~/.conda/envs/wan/bin/python run_experiments.py --sequential
```

## 常见问题

### Q: 如何修改FlowAlign参数？
A: 修改 `DEFAULT_FLOWALIGN_PARAMS` 字典，所有实验自动应用。

### Q: 如何使用不同的strength/guidance_scale？
A: 暂时修改默认参数，或在代码中为特定实验覆盖参数。

### Q: 遇到OOM错误怎么办？
A: 使用 `--sequential` 顺序运行，或检查是否有僵尸进程占用GPU。

### Q: 如何查看详细日志？
A: 脚本会打印最后500字符的输出。完整日志可重定向：
```bash
python run_experiments.py 2>&1 | tee experiment.log
```

### Q: 如何自定义结果保存位置？
A: 使用 `--results-dir` 参数：
```bash
python run_experiments.py --results-dir /path/to/custom/results
```

## 优势

相比手动运行：

| 方面 | 手动运行 | 脚本运行 |
|------|---------|---------|
| 配置管理 | 每次手动编写YAML | 自动生成，单一数据源 |
| GPU选择 | 手动检查nvidia-smi | 自动查找可用GPU |
| 路径处理 | 容易写错 | 自动拼接正确路径 |
| 批量运行 | 逐个启动 | 并行/顺序一键运行 |
| 错误处理 | 手动排查 | 统一错误提示 |
| 可维护性 | 修改N个YAML | 修改一处代码 |

## 未来改进

- [ ] 支持参数扫描（不同guidance_scale）
- [ ] 实时显示推理进度
- [ ] 自动生成实验报告
- [ ] 支持断点续传
- [ ] 集成视频质量评估
