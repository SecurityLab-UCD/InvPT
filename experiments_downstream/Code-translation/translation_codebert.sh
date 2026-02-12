#!/bin/bash
# Downstream evaluation: Code-translation with codebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Code-translation"
./run.sh microsoft/codebert-base "$ROOT_DIR/results/codebert/Code-translation"
