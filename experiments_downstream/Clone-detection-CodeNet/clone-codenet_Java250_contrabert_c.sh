#!/bin/bash
# Downstream evaluation: Clone-detection-CodeNet (Java250) with contrabert_c
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-CodeNet"
./run.sh "$ROOT_DIR/saved_models/ContraBERT_C" "$ROOT_DIR/results/contrabert_c/Clone-detection-CodeNet" Java250 roberta microsoft/codebert-base
