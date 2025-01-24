# PIA
Program-Invariant-Aware Training for Large Language Models in Code Understanding

## Setup
Create virtual environment before running the experiment.
Under the root of project (PIA), do command
```
conda create -n pia python=3.10.12
```

The dataset for our experiment is from CodeSearchNet. You can create a dataset directory yourself and download the Python's jsonline files from [here](https://www.kaggle.com/datasets/omduggineni/codesearchnet).


## Transformation Rules
Supported rules' name are listed below:
- LocalVarRenaming
- AddAssignment2EqualAssignment
- ReverseIfElse


## Instruction
Run the following command to generate a new Jsonl file:
```
python transform.py [RuleName] [Root File] [Output File]

python transform.py LocalVarRenaming dataset/python_train_1.jsonl output/python_train_1_transformed.jsonl
```

