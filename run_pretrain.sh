python modeling/train_roberta.py \
    --batch_size=32 \
    --dataset_path="data/codesearchnet_java.jsonl" \
    --num_train_epochs=20 \
    --run_name="bt_pretrain_20epoch" \
    --contra_type="barlow_twins" \
    --seed=0 
    