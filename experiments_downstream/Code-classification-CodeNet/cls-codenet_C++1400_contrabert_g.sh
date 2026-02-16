#!/bin/bash
# Downstream evaluation: Code-classification-CodeNet (C++1400) with contrabert_g
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Code-classification-CodeNet"
./run.sh "$ROOT_DIR/saved_models/ContraBERT_G" "$ROOT_DIR/results/contrabert_g/Code-classification-CodeNet" C++1400 roberta microsoft/graphcodebert-base
