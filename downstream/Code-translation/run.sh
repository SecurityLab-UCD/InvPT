#!/bin/bash
# run.sh 

# Example: microsoft/codebert-base
pretrained_model=$1
# Example: /mnt/sda/user/ericxu/saved_models/codebert
output_dir=$2
log_path=$output_dir/test.log

# Clean up
rm -f $log_path

echo "Finetuning"
python code/run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path $pretrained_model \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename data/train.java-cs.txt.java,data/train.java-cs.txt.cs \
	--dev_filename data/valid.java-cs.txt.java,data/valid.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000

# No need to keep outputs of finetuning, might be confusing
rm -f $output_dir/*.output

echo
echo "Inference: Java to CS"
python code/run.py \
    --do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--test_filename data/test.java-cs.txt.java,data/test.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 

# Inference output is test_<id>.output, with ID starting at 0
mv $output_dir/test_0.output $output_dir/java_to_cs.output
mv $output_dir/test_0.gold $output_dir/java_to_cs.gold

echo "Java to CS:" >> $log_path
python evaluator/evaluator.py -ref data/test.java-cs.txt.cs -pre $output_dir/java_to_cs.output >> $log_path

echo
echo "Inference: CS to Java"
python code/run.py \
    --do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--test_filename data/test.java-cs.txt.cs,data/test.java-cs.txt.java \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 

# Inference output is test_<id>.output, with ID starting at 0
mv $output_dir/test_0.output $output_dir/cs_to_java.output
mv $output_dir/test_0.gold $output_dir/cs_to_java.gold

echo "CS to Java:" >> $log_path
python evaluator/evaluator.py -ref data/test.java-cs.txt.cs -pre $output_dir/cs_to_java.output >> $log_path
