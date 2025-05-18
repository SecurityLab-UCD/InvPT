#!/bin/bash
# uv run bash run.sh 

# The absolute path to save all the artifacts/result:
outdir=${1:-"$PIA_HOME/downstream/Clone-detection-BCB-naturalness-attack/output"}
echo "All outputs and artifacts are being saved to $outdir"
mkdir -p $outdir

cd ./CodeXGLUE/Clone-detection-BigCloneBench

cd code
echo
echo "Finetune stage"
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

echo
echo "Inference stage"
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

echo
echo "Getting substitutes"
cd dataset
python get_substitutes.py \
    --store_path ./test_subs_test_sampled.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./test_sampled.txt \
    --block_size 512 \
    --index 0 4000

cd ..

echo
echo "Greedy attack"
cd code
# eval_data_file is the attacked subset
python attack.py \
    --output_dir="$outdir/saved_models" \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path "$outdir/attack_base_result.csv" \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee "$outdir/attack.log"
cd ..

#echo
#echo "GA Attack"
#cd code
#python attack.py \
#    --output_dir="$outdir/saved_models" \
#    --model_type=roberta \
#    --config_name=microsoft/codebert-base \
#    --csv_store_path "$outdir/attack_base_result_GA.csv" \
#    --model_name_or_path=microsoft/codebert-base \
#    --tokenizer_name=roberta-base \
#    --use_ga \
#    --base_model=microsoft/codebert-base-mlm \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/test_sampled.txt \ # Attacked subset
#    --test_data_file=../dataset/test_sampled.txt \
#    --block_size 512 \
#    --eval_batch_size 32 \
#    --seed 123456 2>&1| tee "$outdir/attack_GA.log"
#cd ..
#
#echo
#echo "MHM attack"
#cd code
#python mhm_attack.py \
#    --output_dir="$outdir/saved_models" \
#    --model_type=roberta \
#    --tokenizer_name=microsoft/codebert-base \
#    --model_name_or_path=microsoft/codebert-base \
#    --csv_store_path "$outdir/attack_original_mhm.csv" \
#    --original \
#    --base_model=microsoft/codebert-base-mlm \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/test_sampled.txt \ # Attacked subset
#    --test_data_file=../dataset/test_sampled.txt \
#    --block_size 512 \
#    --eval_batch_size 64 \
#    --seed 123456  2>&1 | tee "$outdir/attack_original_mhm.log"
