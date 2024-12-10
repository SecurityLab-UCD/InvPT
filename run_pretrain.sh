python modeling/train_roberta.py \
    --batch_size=32 \
    --num_train_epochs=60 \
    --run_name="mlm_all_bt_pretrain" \
    --contra_type="barlow_twins" \
    --seed=0 \
    --resume_from="saved_models/mlm_all_bt_pretrain" 
    