#!/bin/bash
model_path=$1
save_path=$2
model_type=${3:-roberta}
tokenizer_name=${4:-roberta-base}
operator_key=${5:-}

output_dir=$save_path

mkdir -p $output_dir

if [ -n "$operator_key" ]; then
    test_file=./dataset/aug_test_${operator_key}.jsonl
    prediction_file=aug_predictions_${operator_key}.jsonl
    train_log=$output_dir/aug_train_${operator_key}.log
    eval_log=$output_dir/aug_test_${operator_key}.log
else
    test_file=./dataset/aug_test.jsonl
    prediction_file=aug_predictions.jsonl
    train_log=$output_dir/aug_train.log
    eval_log=$output_dir/aug_test.log
fi

if [ ! -f "$test_file" ]; then
    echo "ERROR: test file not found: $test_file"
    exit 1
fi

touch $train_log

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=$model_type \
    --model_name_or_path=$model_path \
    --tokenizer_name=$tokenizer_name \
    --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=$test_file \
    --test_predictions_file=$prediction_file \
    --epoch 2 \
    --block_size 400 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee $train_log

echo "Extracting answers..."
python evaluator/extract_answers.py \
    -c dataset/test.jsonl \
    -o $output_dir/answer.jsonl


echo "Evaluating test predictions..."
python evaluator/evaluator.py \
    -a $output_dir/answer.jsonl \
    -p $output_dir/$prediction_file > $eval_log

echo "Done"
