#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="PIA"

RUN_NAME="InvGraphCodeBERT"
python -m modeling.pretrain \
    --batch_size=64 \
    --num_epochs=3 \
    --model_name="microsoft/graphcodebert-base" \
    --dataset_path="data/csn.jsonl" \
    --run_name=$RUN_NAME \
    --seed=0 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-5
