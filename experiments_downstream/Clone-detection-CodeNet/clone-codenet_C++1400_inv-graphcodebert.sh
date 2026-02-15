#!/bin/bash
# Downstream evaluation: Clone-detection-CodeNet (C++1400) with inv-graphcodebert
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Clone-detection-CodeNet"
./run.sh "$ROOT_DIR/saved_models/InvGraphCodeBERT-supcon" "$ROOT_DIR/results/inv-graphcodebert/Clone-detection-CodeNet" C++1400 roberta microsoft/graphcodebert-base
