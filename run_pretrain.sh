python modeling/train_roberta.py \
    --batch_size=32 \
    --dataset_path="data/codesearchnet_java.jsonl" \
    --num_train_epochs=2 \
    --run_name="bt_pretrain" \
    --contra_type="barlow_twins" \
    --seed=0 
    