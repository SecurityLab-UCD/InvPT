#!/bin/bash
# uv run bash run.sh 


# Argument processing

USAGE=$(cat <<EOF
SUMMARY

uv run bash test.sh [options] [outdir]

ARGUMENTS
    outdir - the path of the stored model weights ("saved_models")

OPTIONS
    --do_all - Set all options
    --do_finetune - Finetune the model for Clone detection, logs to train.log
    --do_baseline - Run baseline clone detection (without attacks), logs to test.log
    --do_substitute - Get substitutes for attacks
    --do_greedy_attack - Logs to attack.log
    --do_ga_attack - Logs to attack_GA.log
    --do_mhm_attack - Logs to attack_original_mhm.log
EOF
)

if [[ $# == 0 ]]; then
    echo "$USAGE"
    exit 0
fi

outdir=""
do_finetune=0
do_baseline=0
do_greedy_attack=0
do_ga_attack=0
do_mhm_attack=0
do_substitute=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "$USAGE"
            exit 0
            ;;
        --do_all)
            do_finetune=1
            do_baseline=1
            do_greedy_attack=1
            do_ga_attack=1
            do_mhm_attack=1
            do_substitute=1
            shift 1
            ;;
        --do_substitute)
            do_substitute=1
            shift 1
            ;;
        --do_finetune)
            do_finetune=1
            shift 1
            ;;
        --do_baseline)
            do_baseline=1
            shift 1
            ;;
        --do_greedy_attack)
            do_greedy_attack=1
            shift 1
            ;;
        --do_ga_attack)
            do_ga_attack=1
            shift 1
            ;;
        --do_mhm_attack)
            do_mhm_attack=1
            shift 1
            ;;
        *)
            if [[ -n "$outdir" ]]; then
                echo "unrecognized argument: $1"
                exit 1
            fi
            outdir=$1
            shift 1
            ;;
    esac
done

outdir=${outdir:-"$PIA_HOME/attack/Clone-detection-BCB-naturalness-attack/saved_models"}

echo "Configuration:"
echo "outdir: $outdir"
echo "Do finetune: $do_finetune"
echo "Do baseline test: $do_baseline"
echo "Do substitution: $do_substitute"
echo "Do greedy attack: $do_greedy_attack"
echo "Do ga attack: $do_ga_attack"
echo "Do mhm attack: $do_mhm_attack"


# Initialize environment

mkdir -p $outdir
cd ./CodeXGLUE/Clone-detection-BigCloneBench


# Run the pipeline

if (( $do_finetune )); then
cd code
echo
echo "Finetune stage"
python run.py \
    --output_dir="$outdir" \
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
fi

if (( $do_baseline )); then
echo
echo "Inference stage"
cd code
python run.py \
    --output_dir="$outdir" \
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
fi

if (( $do_substitute )); then
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
fi

if (( $do_greedy_attack )); then
echo
echo "Greedy attack"
cd code
# eval_data_file is the attacked subset
python attack.py \
    --output_dir="$outdir" \
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
fi

if (( $do_ga_attack )); then
echo
echo "GA Attack"
cd code
python attack.py \
    --output_dir="$outdir" \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path "$outdir/attack_base_result_GA.csv" \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee "$outdir/attack_GA.log"
cd ..
fi

if (( $do_mhm_attack )); then
echo
echo "MHM attack"
cd code
python mhm_attack.py \
    --output_dir="$outdir" \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path "$outdir/attack_original_mhm.csv" \
    --original \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee "$outdir/attack_original_mhm.log"
fi
