#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="PIA"

    # --model_name="microsoft/graphcodebert-base" \
RUN_NAME="InvContraBERT_G"
python -m modeling.pretrain \
    --batch_size=64 \
    --num_epochs=3 \
    --model_name="./saved_models/ContraBERT_G" \
    --tokenizer_name="microsoft/graphcodebert-base" \
    --dataset_path="data/csn.jsonl" \
    --run_name=$RUN_NAME \
    --seed=0 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-5 \
    --alpha=1.0 \
    --temperature=0.1 \
    --max_seq_length=512 \
    --sample_rate=0.2
