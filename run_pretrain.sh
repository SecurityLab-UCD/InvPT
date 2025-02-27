# total batch size = 
python modeling/train_roberta.py \
    --batch_size=128 \
    --max_steps=100000 \
    --model_name="microsoft/codebert-base" \
    --dataset_path="data/codesearchnet_jp.jsonl" \
    --run_name="jp_nl_all_100k" \
    --seed=0 \
    --percentage=1 \
    --use_nl \
    --gradient_accumulation_steps=2 \
 

       