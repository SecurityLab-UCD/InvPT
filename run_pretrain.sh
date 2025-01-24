python modeling/train_roberta.py \
    --batch_size=128 \
    --model_name="microsoft/codebert-base" \
    --dataset_path="data/codesearchnet_java.jsonl" \
    --num_train_epochs=20 \
    --run_name="bi_encoder_20epoch" \
    --seed=0 
    