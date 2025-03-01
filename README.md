# PIA

Program-Invariant-Aware Training for Large Language Models in Code Understanding

## Training

### Pre-training Datset

We use the transformed CodeSearchNet_java for now.
The dataset can be found in the [artifacts](https://zenodo.org/records/5376257#.YTC3oI4zZsY) of paper "Bridging Pre-trained Models and Downstream Tasks for Source Code Understanding" under "Code search".

```sh
cd data
curl "https://zenodo.org/records/5376257/files/Code%20search.7z?download=1" --output codesearch.7z
7za x codesearch.7z
python3 strip.py
```

### Pre-training RoBERTa

1. Install required packages:

```sh
conda create -n bert python=3.10.12
conda activate bert
pip install -r requirements.txt
```

2. Train the model

```bash
python modeling/train_roberta.py \
    --dataset_path="data/codesearchnet_java.jsonl" \
    --batch_size=32 \
    --num_train_epochs=30 \
    --run_name="ContraBERT_java"
```
