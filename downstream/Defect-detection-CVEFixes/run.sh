model_path=$1
output_dir=$2
dataset_path=dataset/
mkdir -p $output_dir
touch $output_dir/train.log
python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --model_name_or_path=$model_path \
    --tokenizer_name=roberta-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$dataset_path/train.jsonl \
    --eval_data_file=$dataset_path/valid.jsonl \
    --test_data_file=$dataset_path/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee $output_dir/train.log

python evaluator/evaluator.py -a $dataset_path/test.jsonl -p $output_dir/predictions.txt > $output_dir/test.log
