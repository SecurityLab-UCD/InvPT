# total batch size = 
python modeling/train_roberta.py \
    --batch_size=256 \
    --max_steps=50000 \
    --model_name="microsoft/graphcodebert-base" \
    --dataset_path="data/csn_jp.jsonl" \
    --run_name="GraphCodeBERT_JP" \
    --seed=0 \
    --percentage=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-4 \
    --continue_from_pretrained
 

       