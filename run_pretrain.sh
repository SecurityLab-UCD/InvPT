# total batch size = 
python modeling/train_roberta.py \
    --batch_size=256 \
    --max_steps=100000 \
    --model_name="microsoft/codebert-base" \
    --dataset_path="data/codesearchnet_java.jsonl" \
    --run_name="java_5e-5" \
    --seed=0
    