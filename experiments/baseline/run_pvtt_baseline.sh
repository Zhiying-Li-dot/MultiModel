#!/bin/bash
# Run PVTT baseline experiment
# FlowEdit + WAN2.1 for bracelet to necklace transfer

set -e

echo "=== PVTT Baseline Experiment ==="
echo "Task: Replace bracelets with necklace"
echo ""

cd flowedit-wan

# Create results directory
mkdir -p results/pvtt

# Run FlowAlign (recommended)
echo "[1/2] Running FlowAlign..."
python awesome_wan_editing.py --config=./config/pvtt/bracelet_to_necklace.yaml

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to: flowedit-wan/results/pvtt/"
echo ""
echo "Output video: flowedit-wan/results/pvtt/flowalign_bracelet_to_necklace.mp4"
