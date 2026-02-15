#!/bin/bash
# Downstream evaluation: Code-classification-CodeNet (C++1400) with inv-contrabert_g
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Code-classification-CodeNet"
./run.sh "$ROOT_DIR/saved_models/InvContraBERT_G-supcon/final" "$ROOT_DIR/results/inv-contrabert_g/Code-classification-CodeNet" C++1400 roberta microsoft/graphcodebert-base
