# Clone Detection (BCB)

Most information are from the official [CodeXGLUE
repository](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)

## Getting the dataset

```bash
cd dataset
python3 get_bcb.py
```

`original_data.jsonl` is the original `data.jsonl` file from the CodeXGLUE
repository. `augmented_data.jsonl` is the augmented version.

## Evaluator

```bash
python evaluator/evaluator.py -a evaluator/answers.txt -p evaluator/predictions.txt
```

{'Recall': 0.25, 'Precision': 0.5, 'F1': 0.3333333333333333}

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```b
13653451	21955002	0
1188160	8831513	1
1141235	14322332	0
16765164	17526811	1
```

## Running the pipelines

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run.sh <model_path_or_name> <save_path>

# for example,
CUDA_VISIBLE_DEVICES=0,1 ./run.sh microsoft/codebert-base saved_models/bcb-codebert
```

The evaluation results for the original dataset and the augmented dataset are in
`<save_path>/test.log` and `<save_path>/aug_test.log` respectively.
