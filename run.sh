#!/usr/bin/env bash
set -euo pipefail

uv run experiments_downstream/run_all_downstream.py --all --loss supcon
