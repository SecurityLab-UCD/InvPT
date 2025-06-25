model_path=$1
save_path=$2
subset=$3

output_dir=$save_path/$subset

mkdir -p $output_dir
touch $output_dir/test_train.log

echo "Running fine-tuning for POJ104"
python ./code/run.py \
    --output_dir=$output_dir \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=$model_path \
    --do_train \
    --do_test \
    --train_data_file=./dataset/$subset/train.jsonl \
    --eval_data_file=./dataset/$subset/valid.jsonl \
    --test_data_file=./dataset/$subset/test.jsonl \
    --num_train_epochs 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee $output_dir/test_train.log

echo "Running evaluation for augmented test set..."
./run_aug_test.sh $model_path $save_path $subset