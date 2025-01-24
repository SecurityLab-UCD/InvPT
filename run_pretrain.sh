python modeling/train_roberta.py \
    --batch_size=256 \
    --model_name="microsoft/codebert-base" \
    --dataset_path="data/codesearchnet_java.jsonl" \
    --num_train_epochs=40 \
    --run_name="bi_encoder_40epoch" \
    --seed=0 
    