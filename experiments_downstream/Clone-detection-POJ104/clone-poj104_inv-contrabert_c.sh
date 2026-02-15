#!/bin/bash
# Downstream evaluation: Clone-detection-POJ104 with inv-contrabert_c
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Clone-detection-POJ104"
./run.sh "$ROOT_DIR/saved_models/InvContraBERT_C-supcon/final" "$ROOT_DIR/results/inv-contrabert_c/Clone-detection-POJ104" roberta microsoft/codebert-base
