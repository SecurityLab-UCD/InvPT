#!/bin/bash
model_path=$1
save_path=$2
subset=$3

output_dir=$save_path/$subset

mkdir -p $output_dir
touch $output_dir/train.log

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --model_name_or_path=$model_path \
    --tokenizer_name=roberta-base \
    --do_train \
    --do_test \
    --train_data_file=./dataset/$subset/train.jsonl \
    --eval_data_file=./dataset/$subset/valid.jsonl \
    --test_data_file=./dataset/$subset/test.jsonl \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee $output_dir/train.log

# test
python evaluator/extract_answers.py \
    -c dataset/$subset/test.jsonl \
    -o $output_dir/answer.jsonl

python evaluator/evaluator.py \
    -a $output_dir/answer.jsonl \
    -p $output_dir/predictions.jsonl > $output_dir/test.log


echo "Running evaluation for augmented test set..."
./run_aug_test.sh $model_path $save_path $subset
