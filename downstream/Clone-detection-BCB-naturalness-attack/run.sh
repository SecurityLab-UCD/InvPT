#!/bin/bash
# uv run bash run.sh 

# The absolute path to save all the artifacts/result:
outdir=${1:-"$PIA_HOME/downstream/Clone-detection-BCB-naturalness-attack/output"}
echo "All outputs and artifacts are being saved to $outdir"
mkdir -p $outdir

cd ./CodeXGLUE/Clone-detection-BigCloneBench

cd code
echo "Finetuning"
python run.py \
    --output_dir="$outdir/saved_models/" \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee "$outdir/train.log"
cd ..

echo "Inference"
cd code
python run.py \
    --output_dir="$outdir/saved_models" \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_test \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee "$outdir/test.log"
cd ..
