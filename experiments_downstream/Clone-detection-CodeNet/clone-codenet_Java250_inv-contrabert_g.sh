#!/bin/bash
# Downstream evaluation: Clone-detection-CodeNet (Java250) with inv-contrabert_g
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-CodeNet"
./run.sh "$ROOT_DIR/saved_models/InvContraBERT_G-supcon/final" "$ROOT_DIR/results/inv-contrabert_g/Clone-detection-CodeNet" Java250 roberta microsoft/graphcodebert-base
