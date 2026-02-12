#!/bin/bash
# Downstream evaluation: Clone-detection-CodeNet (Java250) with inv-codebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-CodeNet"
./run.sh "$ROOT_DIR/saved_models/InvCodeBERT-supcon" "$ROOT_DIR/results/inv-codebert/Clone-detection-CodeNet" Java250 roberta microsoft/codebert-base
