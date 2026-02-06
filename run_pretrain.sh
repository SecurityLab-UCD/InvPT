#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="InvPT"

# Use a YAML config; override specific values with CLI options if needed:
#   python modeling/cli.py run experiments/base.yaml --seed 42
python modeling/cli.py run experiments/base.yaml
