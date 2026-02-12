#!/bin/bash
# Downstream evaluation: Clone-detection-BigCloneBench with inv-graphcodebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Clone-detection-BigCloneBench"
./run.sh "$ROOT_DIR/saved_models/InvGraphCodeBERT-supcon" "$ROOT_DIR/results/inv-graphcodebert/Clone-detection-BigCloneBench"
