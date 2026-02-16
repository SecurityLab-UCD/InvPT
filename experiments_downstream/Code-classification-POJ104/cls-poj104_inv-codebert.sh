#!/bin/bash
# Downstream evaluation: Code-classification-POJ104 with inv-codebert
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Code-classification-POJ104"
./run.sh "$ROOT_DIR/saved_models/InvCodeBERT-supcon" "$ROOT_DIR/results/inv-codebert/Code-classification-POJ104" "" roberta microsoft/codebert-base
