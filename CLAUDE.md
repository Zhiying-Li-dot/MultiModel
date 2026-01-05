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
