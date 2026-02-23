#!/bin/bash
# Downstream evaluation: Code-classification-CodeNet (Python800) with codesage
set -euo pipefail

# Parse CUDA device argument (default: 0)
CUDA_DEVICE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR/downstream/Code-classification-CodeNet"
./run.sh codesage/codesage-small "$ROOT_DIR/results/codesage/Code-classification-CodeNet" Python800 codesage codesage/codesage-small
