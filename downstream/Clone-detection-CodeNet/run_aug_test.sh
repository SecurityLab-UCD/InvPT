model_path=$1
save_path=$2
subset=$3

output_dir=$save_path/$subset

mkdir -p $output_dir
touch $output_dir/aug_train.log

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=$model_path \
    --tokenizer_name=roberta-base \
    --do_test \
    --train_data_file=./dataset/$subset/train.jsonl \
    --eval_data_file=./dataset/$subset/valid.jsonl \
    --test_data_file=./dataset/$subset/aug_test.jsonl \
    --test_predictions_file=aug_predictions.jsonl \
    --epoch 2 \
    --block_size 400 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee $output_dir/aug_train.log

echo "Extracting answers..."
python evaluator/extract_answers.py \
    -c dataset/$subset/test.jsonl \
    -o $output_dir/answer.jsonl


echo "Evaluating test predictions..."
python evaluator/evaluator.py \
    -a $output_dir/answer.jsonl \
    -p $output_dir/aug_predictions.jsonl > $output_dir/aug_test.log

echo "Done"