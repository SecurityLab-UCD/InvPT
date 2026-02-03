# CodeNet Clone Detection

## Task Definition

Given a code and a collection of candidates as the input, the task is to return Top K codes with the same semantic.
Models are evaluated by MAP@R score.
MAP@R is defined as the mean of average precision scores, each of which is evaluated for retrieving R most similar samples given a query.
For a code (query), R is the number of other codes in the same class.

For example, Java250 consists of 250 problems, where each includes 300 Java programs.
Therefore, R=299 in this task (1 query, 299 results).

## Dataset

We use [CodeNet](https://github.com/IBM/Project_CodeNet) dataset on this task.

### Download and Preprocess

```bash
cd dataset
python get_codenet.py <dataset>

# or step by step
# this will write data in ./raw/
python download.py <dataset>
python preprocess.py <dataset>
```

The pre-processed dataset will be saved as `<dataset>.jsonl`,
following the same format as POJ-104 for clone detection.

`<dataset>` is one of the following:

- Java250
- Python800
- C++1000
- C++1400
- all

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function. One row is illustrated below.

- **code:** the source code
- **label:** the number of problem that the source code solves
- **index:** the index of example

### Data Statistics

Split ratio is 50%, 25%, 25% for train, valid, test.

## Evaluation and Fine-tuning

These scripts are directly copied from `../Clone-detection-POJ-104`,
please refer to the documentation there.

## Running the Pipelines

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run.sh <model_path_or_name> <save_path> <subset>

# for example,
CUDA_VISIBLE_DEVICES=0,1 ./run.sh microsoft/codebert-base saved_models/codebert-Java250 Java250
```
