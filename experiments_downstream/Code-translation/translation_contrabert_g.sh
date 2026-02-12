#!/bin/bash
# Downstream evaluation: Code-translation with contrabert_g
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Code-translation"
./run.sh "$ROOT_DIR/saved_models/ContraBERT_G" "$ROOT_DIR/results/contrabert_g/Code-translation"
