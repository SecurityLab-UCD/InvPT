#!/bin/bash
# Downstream evaluation: Code-classification-CodeNet (C++1400) with contrabert_c
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/downstream/Code-classification-CodeNet"
./run.sh "$ROOT_DIR/saved_models/ContraBERT_C" "$ROOT_DIR/results/contrabert_c/Code-classification-CodeNet" C++1400 roberta microsoft/codebert-base
