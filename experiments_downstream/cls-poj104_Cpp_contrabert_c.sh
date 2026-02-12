#!/bin/bash
# Downstream evaluation: Code-classification-POJ104 (Cpp) with contrabert_c
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/downstream/Code-classification-POJ104"
./run.sh "$ROOT_DIR/saved_models/ContraBERT_C" "$ROOT_DIR/results/contrabert_c/Code-classification-POJ104" Cpp roberta microsoft/codebert-base
