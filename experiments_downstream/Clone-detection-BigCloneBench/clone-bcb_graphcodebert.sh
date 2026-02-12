#!/bin/bash
# Downstream evaluation: Clone-detection-BigCloneBench with graphcodebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-BigCloneBench"
./run.sh microsoft/graphcodebert-base "$ROOT_DIR/results/graphcodebert/Clone-detection-BigCloneBench"
