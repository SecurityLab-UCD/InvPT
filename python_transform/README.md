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
python transform.py -t [RuleName] -i [Input File] -o [Output File] -n [Number of CPUs (optional, defaults to all available CPUs)]
```

Examples: The command below applies the LocalVarRenaming rule to the input file dataset/python_train_1.jsonl and saves the transformed output to output/python_train_1_transformed.jsonl:

```
python transform.py  -t LocalVarRenaming -i dataset/python_train_1.jsonl -o output/python_train_1_transformed.jsonl
```