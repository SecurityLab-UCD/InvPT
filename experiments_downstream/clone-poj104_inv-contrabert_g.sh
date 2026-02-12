#!/bin/bash
# Downstream evaluation: Clone-detection-POJ104 with inv-contrabert_g
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-POJ104"
./run.sh "$ROOT_DIR/saved_models/InvContraBERT_G-supcon" "$ROOT_DIR/results/inv-contrabert_g/Clone-detection-POJ104" roberta microsoft/graphcodebert-base
