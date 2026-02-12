#!/bin/bash
# Downstream evaluation: Clone-detection-CodeNet (Python800) with codebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-CodeNet"
./run.sh microsoft/codebert-base "$ROOT_DIR/results/codebert/Clone-detection-CodeNet" Python800 roberta microsoft/codebert-base
