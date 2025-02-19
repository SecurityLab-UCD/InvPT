# total batch size = 
python modeling/train_roberta.py \
    --batch_size=256 \
    --max_steps=10000 \
    --model_name="microsoft/codebert-base" \
    --dataset_path="data/codesearchnet_jp.jsonl" \
    --run_name="jp_nl_20p_10k" \
    --seed=0 \
    --percentage=0.2
    