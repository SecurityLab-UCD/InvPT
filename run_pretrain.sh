export CUDA_VISIBLE_DEVICES=4,5,6,7 

# stage 1
python modeling/train_stage1.py \
    --batch_size=256 \
    --max_steps=20000 \
    --model_name="microsoft/graphcodebert-base" \
    --dataset_path="data/raw_csn.jsonl" \
    --run_name="InvariantBERT_G-stage1" \
    --seed=0 \
    --gradient_accumulation_steps=1 \
    --learning_rate=4e-4 


# total batch size = 
python modeling/train_roberta.py \
    --batch_size=256 \
    --max_steps=50000 \
    --model_name="microsoft/graphcodebert-base" \
    --dataset_path="data/csn_jp.jsonl" \
    --run_name="InvariantBERT_G-stage2" \
    --seed=0 \
    --percentage=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-4 \
    --continue_from_pretrained
 

       