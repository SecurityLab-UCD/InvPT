#!/bin/bash
# Downstream evaluation: Defect-detection with graphcodebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Defect-detection"
./run.sh microsoft/graphcodebert-base "$ROOT_DIR/results/graphcodebert/Defect-detection"
