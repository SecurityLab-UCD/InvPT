#!/bin/bash
# Downstream evaluation: Defect-detection with inv-codebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Defect-detection"
./run.sh "$ROOT_DIR/saved_models/InvCodeBERT-supcon" "$ROOT_DIR/results/inv-codebert/Defect-detection"
