#!/bin/bash
# Downstream evaluation: Code-translation with inv-codebert
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Code-translation"
./run.sh "$ROOT_DIR/saved_models/InvCodeBERT-supcon/final" "$ROOT_DIR/results/inv-codebert/Code-translation"
