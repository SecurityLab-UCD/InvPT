#!/bin/bash
# Downstream evaluation: Clone-detection-POJ104 with graphcodebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-POJ104"
./run.sh microsoft/graphcodebert-base "$ROOT_DIR/results/graphcodebert/Clone-detection-POJ104" roberta microsoft/graphcodebert-base
