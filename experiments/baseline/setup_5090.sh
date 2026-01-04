#!/bin/bash
# Setup script for PVTT baseline experiments on 5090 machine
# FlowEdit + WAN2.1 baseline

set -e

echo "=== PVTT Baseline Setup for 5090 ==="

# 1. Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n wanalign python=3.10 -y
conda activate wanalign

# 2. Install PyTorch (adjust CUDA version if needed)
echo "[2/4] Installing PyTorch..."
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
pip install matplotlib omegaconf imageio
pip install transformers==4.51.3 accelerate
pip install imageio[ffmpeg] ftfy

# 4. Install custom diffusers
echo "[4/4] Installing custom diffusers..."
cd flowedit-wan/diffusers
pip install -e .
cd ../..

# Create results directory
mkdir -p flowedit-wan/results/pvtt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run PVTT baseline experiment:"
echo "  conda activate wanalign"
echo "  cd flowedit-wan"
echo "  python awesome_wan_editing.py --config=./config/pvtt/bracelet_to_necklace.yaml"
echo ""
echo "Results will be saved to: flowedit-wan/results/pvtt/"
