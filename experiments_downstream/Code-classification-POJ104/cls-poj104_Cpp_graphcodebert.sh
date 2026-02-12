#!/bin/bash
# Downstream evaluation: Code-classification-POJ104 (Cpp) with graphcodebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Code-classification-POJ104"
./run.sh microsoft/graphcodebert-base "$ROOT_DIR/results/graphcodebert/Code-classification-POJ104" Cpp roberta microsoft/graphcodebert-base
