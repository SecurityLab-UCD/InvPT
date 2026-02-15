#!/bin/bash
# Downstream evaluation: Clone-detection-CodeNet (Java250) with inv-modernbert
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Clone-detection-CodeNet"
./run.sh "$ROOT_DIR/saved_models/aug-only/InvModernBERT-supcon/final" "$ROOT_DIR/results/inv-modernbert/Clone-detection-CodeNet" Java250 modernbert answerdotai/ModernBERT-base
