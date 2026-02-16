#!/bin/bash
model_path=$1
save_path=$2
subset=$3
model_type=${4:-roberta}
tokenizer_name=${5:-microsoft/codebert-base}

base_name=$(basename "$save_path")
if [ "$base_name" = "$subset" ]; then
    output_dir=$save_path
else
    output_dir=$save_path/$subset
fi

mkdir -p $output_dir
touch $output_dir/test_train.log

train_suffix=""
if [ -n "$subset" ]; then
    train_suffix="/$subset"
fi

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=$model_type \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_path \
    --do_test \
    --train_data_file=./dataset${train_suffix}/train.jsonl \
    --eval_data_file=./dataset${train_suffix}/valid.jsonl \
    --test_data_file=./dataset${train_suffix}/aug_test.jsonl \
    --num_train_epochs 5 \
    --block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee $output_dir/aug_test.log
