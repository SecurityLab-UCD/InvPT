export CUDA_VISIBLE_DEVICES=4,5,6,7 
export WANDB_PROJECT="PIA"

RUN_NAME="InvariantBERT_G_self_contrast"
python modeling/pretrain.py \
    --batch_size=256 \
    --max_steps=50000 \
    --model_name="microsoft/graphcodebert-base" \
    --dataset_path="data/csn_jp.jsonl" \
    --run_name=$RUN_NAME \
    --seed=0 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5

