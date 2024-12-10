model_path=$1
output_dir=$2
mkdir -p $output_dir
touch $output_dir/train.log
python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=$model_path \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee $output_dir/train.log