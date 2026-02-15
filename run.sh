#!/usr/bin/env bash
set -euo pipefail

loss="supcon"
models=(
  "inv-codebert"
  "inv-graphcodebert"
  "inv-contrabert_c"
  "inv-contrabert_g"
  "codebert"
  "graphcodebert"
  "contrabert_c"
  "contrabert_g"
  # "inv-modernbert"
  # "modernbert"
)

for model in "${models[@]}"; do
    uv run experiments_downstream/run_all_downstream.py \
      --loss "${loss}" \
      --model "${model}"
done
