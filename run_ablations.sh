#!/usr/bin/env bash
# Run all Tier 1 ablation experiments sequentially on GPUs 4,5,6,7.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

for cfg in experiments/ablation/*.yaml; do
    echo "=========================================="
    echo "Running: $cfg"
    echo "Started: $(date)"
    echo "=========================================="
    accelerate launch --multi_gpu modeling/cli.py run "$cfg"
    echo "Finished: $(date)"
    echo ""
done

echo "All ablation experiments complete."
