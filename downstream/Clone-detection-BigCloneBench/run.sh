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
    --train_data_file=./dataset/train.txt \
    --eval_data_file=./dataset/valid.txt \
    --test_data_file=./dataset/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee $output_dir/train.log

python evaluator/evaluator.py -a dataset/test.txt -p $output_dir/predictions.txt > $output_dir/test.log