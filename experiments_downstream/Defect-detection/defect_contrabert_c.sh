#!/bin/bash
# Downstream evaluation: Defect-detection with contrabert_c
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Defect-detection"
./run.sh "$ROOT_DIR/saved_models/ContraBERT_C" "$ROOT_DIR/results/contrabert_c/Defect-detection"
