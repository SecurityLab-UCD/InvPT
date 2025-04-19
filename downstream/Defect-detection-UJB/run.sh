model_path=$1
tokenizer_name=$2
output_dir=$3
mkdir -p $output_dir
touch $output_dir/train.log

echo "Train and test on devign..."

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_path \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=./dataset/devign/train.jsonl \
    --eval_data_file=./dataset/devign/valid.jsonl \
    --test_data_file=./dataset/devign/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee $output_dir/train.log

python evaluator/evaluator.py -a dataset/devign/test.jsonl -p $output_dir/predictions.txt > $output_dir/devign_test.log

rm -rf $output_dir/predictions.txt

echo "Test on unaugmented ujb..."

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_path \
    --do_test \
    --train_data_file=./dataset/devign/train.jsonl \
    --eval_data_file=./dataset/devign/valid.jsonl \
    --test_data_file=./dataset/ujb/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456

python evaluator/evaluator.py -a dataset/ujb/test.jsonl -p $output_dir/predictions.txt > $output_dir/ujb_test.log

rm -rf $output_dir/predictions.txt

echo "Test on augmented ujb..."

python ./code/run.py \
    --output_dir=$output_dir \
    --model_type=roberta \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_path \
    --do_test \
    --train_data_file=./dataset/devign/train.jsonl \
    --eval_data_file=./dataset/devign/valid.jsonl \
    --test_data_file=./dataset/ujb/aug_test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456

python evaluator/evaluator.py -a dataset/ujb/aug_test.jsonl -p $output_dir/predictions.txt > $output_dir/ujb_aug_test.log
