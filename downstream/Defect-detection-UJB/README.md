# Defect Detection (UJB)

Most of the pipeline is based on CodeXGLUE's Defect detection pipeline for
Devign.

## Task Definition

Given a source code, the task is to identify whether it is an insecure code that
may attack software systems, such as resource leaks, use-after-free
vulnerabilities and DoS attack.  We treat the task as binary classification
(0/1), where 1 stands for insecure code and 0 for secure code.

### Dataset

The finetuning dataset we use comes from the paper [*Devign*: Effective
Vulnerability Identification by Learning Comprehensive Program Semantics via
Graph Neural
Networks](http://papers.nips.cc/paper/9209-devign-effective-vulnerability-identification-by-learning-comprehensive-program-semantics-via-graph-neural-networks.pdf).
We combine all projects and split 80%/10%/10% for training/dev/test.

The testing dataset comes from the paper [CoderUJB: An Executable and Unified
Java Benchmark for Practical Programming
Scenarios](https://arxiv.org/pdf/2403.19287v1).

### Download and Preprocess


```bash
# Downloads finetuning dataset (Devign)
python3 get_devign.py
# Downloads testing dataset (UJB) and augment it (UJB + T)
python3 get_ujb.py
```

### Data Format

#### Devign
After getting the Devign dataset, you can obtain three .jsonl files, i.e.
train.jsonl, valid.jsonl, test.jsonl, stored in `./dataset/devign/`

For each file, each line in the uncompressed file represents one function.  One
row is illustrated below.

- **func:** the source code
   - **target:** 0 or 1 (vulnerability or not)
   - **idx:** the index of example

#### UJB
After getting the UJB dataset, you can obtain two .jsonl files, i.e.
test.jsonl, aug_test.jsonl, stored in `./dataset/ujb/`

The file format is identical to that of Devign. `aug_test.jsonl` is `test.jsonl`
accumulatively augmented.

### Data Statistics

Data statistics of the devign finetuning dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  21,854   |
| Dev   |   2,732   |
| Test  |   2,732   |

There are 940 examples in the UJB dataset.


## Evaluator

We provide a script to evaluate predictions for this task, and report accuracy
score.

### Example

```shell
python evaluator/evaluator.py -a evaluator/test.jsonl -p
evaluator/predictions.txt 
```

{'Acc': 0.6}

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```shell
0	0
1	1
2	1
3	0
4	0
```

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task.

### Fine-tune

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```


### Inference

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
```


### All Togerher

```sh
./run.sh <pretrained_model_path> <tokenizer_name> <output_path>
```

### Evaluation

```shell
python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt
```

{'Acc': 0.6207906295754027}

## Result

The results on the test set are shown as below:

| Methods  |    ACC    |
| -------- | :-------: |
| BiLSTM   |   59.37   |
| TextCNN  |   60.69   |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)  |   61.05   |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) | **62.08** |

## Reference
<pre><code>@inproceedings{zhou2019devign,
  title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
  author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10197--10207},
  year={2019}
}</code></pre>

<pre><code>@inproceedings{zeng2024coderujb,
  title={Coderujb: An executable and unified java benchmark for practical programming scenarios},
  author={Zeng, Zhengran and Wang, Yidong and Xie, Rui and Ye, Wei and Zhang, Shikun},
  booktitle={Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages={124--136},
  year={2024}
}</code></pre>
