#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="InvPT"

# Use torchrun for DDP; uses all visible GPUs (control with CUDA_VISIBLE_DEVICES).
torchrun --nproc_per_node=gpu modeling/cli.py run experiments/base.yaml
