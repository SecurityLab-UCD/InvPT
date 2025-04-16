export CUDA_VISIBLE_DEVICES=4,5,6,7 
export WANDB_PROJECT="PIA"

RUN_NAME="InvariantBERT_G_mix_all_2e-4"
# total batch size = 
python modeling/pretrain.py \
    --batch_size=256 \
    --max_steps=50000 \
    --model_name="microsoft/graphcodebert-base" \
    --dataset_path="data/csn.jsonl" \
    --run_name=$RUN_NAME \
    --seed=0 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-4

