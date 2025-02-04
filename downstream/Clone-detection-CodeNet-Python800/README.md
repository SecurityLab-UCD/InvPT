# Python800 -- Clone Detection (CodeNet)

## Getting the dataset
Download the dataset from https://developer.ibm.com/exchanges/data/all/project-codenet/. 

Or, download by running the following command:
```bash
cd dataset
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz
tar -xvzf Project_CodeNet_Python800.tar.gz
rm Project_CodeNet_Python800.tar.gz 
cd ..
```

## Dataset

The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

unsure

### Data Statistics

Data statistics of the Python 800 dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  120,000  |
| Test  |   60,000  |
| Valid |   60,000  |

Before & After adding transformed code Test Set
|        | #Examples of Test Set |
| -----  | :-------:             |
| Before |   60,000              |
| After  |  118,895              |
## Evaluator