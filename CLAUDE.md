# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PVTT (Product Video Template Transfer) is a research project targeting CVPR 2027. The goal is to generate new product promotional videos by transferring the style, camera movement, and rhythm from a successful template video to new product images.

## Running Baseline Experiments

Experiments run on the 5090 machine via SSH:

```bash
# Use existing wan conda environment
ssh 5090

# Set HuggingFace mirror (required in China)
export HF_ENDPOINT=https://hf-mirror.com

# Run FlowAlign baseline
cd ~/pvtt/baseline/flowedit-wan2.1
~/.conda/envs/wan/bin/python awesome_wan_editing.py --config=./config/pvtt/bracelet_to_necklace.yaml
```

## Creating New Experiment Configs

Add YAML configs to `baseline/flowedit-wan/config/pvtt/`:

```yaml
video:
    video_path: ./videos/your_video.mp4
    source_prompt: Description of source product...
    target_prompt: Description of target product...
    source_blend: source_object_name
    target_blend: target_object_name

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

## Key Directories

- `baseline/flowedit-wan2.1/` - WANAlign2.1 baseline code (Wan2.1)
- `baseline/flowedit-wan2.2/` - FlowAlign with Wan2.2 TI2V-5B
- `experiments/README.md` - Experiment records
- `experiments/results/` - Experiment output videos
- `docs/` - Research plan, literature review, baseline design
- `data/samples/` - Sample product videos and images

## Remote Machine Access

The 5090 machine has 8x RTX 5090 32GB GPUs. Use SSH alias `5090` to connect.

### GPU 选择

运行实验前先检查 GPU 占用情况，**优先选择空闲的 GPU**：

```bash
# 查看 GPU 显存占用
ssh 5090 'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv'

# 使用 CUDA_VISIBLE_DEVICES 指定空闲 GPU
export CUDA_VISIBLE_DEVICES=2  # 选择显存充足的 GPU
```

## Important: Experiment File Paths

**在运行实验之前，必须先查阅配置文件或实验日志获取正确的文件路径！**

- 测试用例数据统一存放在 `data/pvtt-benchmark/cases/{case_name}/`
- 每个 case 包含：`source_video.mp4`, `target_frame1.png`, `config.yaml`
- **不要**自己去 `find` 或猜测文件路径
- **优先**查阅：
  1. `baseline/compositional-flux-ti2v/config/{case_name}.yaml`
  2. `experiments/logs/` 中的实验日志记录的命令

示例（bracelet_to_necklace）：
```bash
CASE_DIR=~/pvtt/data/pvtt-benchmark/cases/bracelet_to_necklace
python scripts/ti2v_rfsolver.py \
    --source-video $CASE_DIR/source_video.mp4 \
    --target-frame $CASE_DIR/target_frame1.png \
    --width 832 --height 480 \
    ...
```

## 实验评估注意事项

### std 不是可靠的质量指标

- **不要**只看 inverted noise 的 std 是否接近 1.0
- std 接近 1.0 不代表视觉质量好（实验证明 33-49 帧 std 最接近 1.0 但质量最差）
- **必须**打开视频查看实际视觉质量

### 视觉质量判断要谨慎

- 不要轻易下"效果好"或"效果差"的结论
- 对比多个帧（不只是第一帧）
- 有疑问时让用户自己判断
