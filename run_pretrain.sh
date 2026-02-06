#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="InvPT"

# Use a YAML config; override specific values with CLI options if needed:
#   python -m modeling run experiments/base.yaml --seed 42
python -m modeling run experiments/base.yaml
