#!/bin/bash
# run.sh 

cd code
# Example: microsoft/codebert-base
pretrained_model=$1
# Example: /mnt/sda/user/ericxu/saved_models/codebert
output_dir=$2
log_path=$output_dir/test.log
aug_log_path=$output_dir/aug_test.log

echo "Finetuning"
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path $pretrained_model \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000

echo
echo "Unaugmented inference"
python run.py \
    --do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 

rm $log_path
echo "Java to CS:" >> $log_path
python evaluator/evaluator.py -ref data/test.java-cs.txt.cs -pre $output_dir/java-cs-model1.output >> $log_path
echo "CS to Java:" >> $log_path
python evaluator/evaluator.py -ref data/test.java-cs.txt.java -pre $output_dir/cs-java-model1.output >> $log_path

echo
echo "Augmented inference"
python run.py \
    --do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--dev_filename ../data/aug_valid.java-cs.txt.java,../data/aug_valid.java-cs.txt.cs \
	--test_filename ../data/aug_test.java-cs.txt.java,../data/aug_test.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 

echo "Java to CS:" >> $aug_log_path
python ../evaluator/evaluator.py -ref ../data/aug_test.java-cs.txt.cs -pre $output_dir/java-cs-model1.output >> $aug_log_path
echo "CS to Java:" >> $aug_log_path
python ../evaluator/evaluator.py -ref ../data/aug_test.java-cs.txt.java -pre $output_dir/cs-java-model1.output >> $aug_log_path
