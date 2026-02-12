#!/bin/bash
# Downstream evaluation: Code-translation with graphcodebert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR/downstream/Code-translation"
./run.sh microsoft/graphcodebert-base "$ROOT_DIR/results/graphcodebert/Code-translation"
