# total batch size = 
python modeling/train_roberta.py \
    --batch_size=256 \
    --max_steps=100000 \
    --model_name="microsoft/codebert-base" \
    --dataset_path="data/codesearchnet_jp.jsonl" \
    --run_name="jp_continue_50k_2e-4" \
    --seed=0 \
    --percentage=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-4 \
    --continue_from_pretrained \
    --resume
 

       