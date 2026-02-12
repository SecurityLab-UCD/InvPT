#!/bin/bash
# Downstream evaluation: Code-classification-CodeNet (C++1400) with codebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Code-classification-CodeNet"
./run.sh microsoft/codebert-base "$ROOT_DIR/results/codebert/Code-classification-CodeNet" C++1400 roberta microsoft/codebert-base
