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

total_models=${#models[@]}
current=0

render_progress() {
  local idx="$1"
  local total="$2"
  local width=30
  local filled=$((idx * width / total))
  local empty=$((width - filled))
  local bar
  bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  bar+="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  printf "[%s] %d/%d" "${bar}" "${idx}" "${total}"
}

for model in "${models[@]}"; do
  current=$((current + 1))
  progress_line=$(render_progress "${current}" "${total_models}")
  printf "%s %s\n" "${progress_line}" "${model}"
  uv run experiments_downstream/run_all_downstream.py \
    --loss "${loss}" \
    --model "${model}"
done
