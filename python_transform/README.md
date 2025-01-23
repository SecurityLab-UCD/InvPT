# PIA
Program-Invariant-Aware Training for Large Language Models in Code Understanding

## Setup
Create virtual environment before running the experiment
```
pip install -r requirements.txt
```

The dataset for our experiment is from CodeSearchNet. You can create a dataset directory yourself and download the Python's jsonline files from [here](https://www.kaggle.com/datasets/omduggineni/codesearchnet).


## Instruction
Run the following command to generate a new Jsonl file:
```
python3.11 transform.py [RuleId] [Root File] [OutputDir]

python3.11 transform.py 1 dataset/python_train_1.jsonl output
```

RuleId stands for one specific transformation method (only ruleIDs 0, 2, and 4 satisfy our research purpose):
- 0 Local Varible Renaming
- 1 Function Definition Reordering
- 2 Reverse If Else Statement
- 3 Statements Order Rearrangement
- 4 Operation Assignment to EqualAssignment
- 5 While to For
- 6 For to While