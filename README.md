# InvPT: Invariant Pre-training for Robust Code Representation Learning

InvPT is a novel pre-training method that improves both the performance and robustness of code representation models against semantically equivalent but syntactically different programs (invariant programs). InvPT applies semantic-preserving code transformations to the pre-training corpus, then continues pre-training state-of-the-art encoder models using a combination of masked language modeling and invariant contrastive learning.

Key design choices:

- **PL-only pre-training**: Removes natural language docstrings, focusing solely on programming language data.
- **Invariant contrastive learning**: Uses InfoNCE or Supervised Contrastive (SupCon) loss between original code and its semantic-preserving transformations with a single shared encoder (no momentum contrast). SupCon mode treats all augmentations of the same function as positives within a batch.
- **Indirect curriculum learning**: Simultaneously trains on self-contrast (easy) and invariant-contrast (hard) examples with dynamic learning rate scheduling.

## Usage

### Environment

#### Python

We use [`uv`](https://docs.astral.sh/uv/guides/install-python/) to manage Python environments.

```sh
uv sync
```

#### Java

We need a Java 11+ JDK for the Java augmentation.
For our experiments, we use OpenJDK 21.

```sh
export JDK_LIB=/usr/lib/jvm/java-21-openjdk-amd64/lib
```

You should modify this line in `.envrc`.

#### C/C++

We use Clang-14 and LLVM-14 for C/C++ augmentation.
We have a script to install the required packages:

```sh
./clang.sh
```

This will install the required packages for C/C++ augmentation to `$HOME/clang+llvm`.
Please make sure to provide this path in `.envrc`:

```sh
export LLVM=$HOME/clang+llvm
export LIBCLANG_PATH=$LLVM/lib/libclang.so
export LD_LIBRARY_PATH=$LLVM/lib:$LD_LIBRARY_PATH
```

#### Loading Environment Variables

Before running the code, please make sure to load the environment variables:

```sh
source .envrc
```

To avoid sourcing the environment variables every time,
we recommend using [`direnv`](https://direnv.net/) to automatically load the environment variables when you enter the directory,
and unload them when you leave the directory.

### Pre-training Dataset

We use CodeSearchNet for pre-training.

```sh
uv run data/get_code_search_net.py
```

This script will download the CodeSearchNet dataset and convert it to JSONL format.
It writes the following files:

- `raw_csn.jsonl`: the entire CodeSearchNet dataset
- `raw_csn_py.jsonl`, `raw_csn_java.jsonl`: the Python and Java subsets of CodeSearchNet

### Data Augmentation

After downloading the dataset, apply invariant code transformations:

```sh
uv run python_transform/augment_pretrain.py data/raw_csn_py.jsonl data/aug_csn_py.jsonl
uv run java_transform/augment_pretrain.py data/raw_csn_java.jsonl data/aug_csn_java.jsonl
```

Then combine original and augmented data for pre-training:

```sh
cp data/raw_csn.jsonl data/csn.jsonl
cat data/aug_csn_py.jsonl >> data/csn.jsonl
cat data/aug_csn_java.jsonl >> data/csn.jsonl
```

The resulting file `data/csn.jsonl` will be used for pre-training.

### Pre-training

```sh
./run_pretrain.sh
```

This script continues pre-training a RoBERTa-based model (e.g., GraphCodeBERT, ContraBERT) with InvPT on 4 GPUs. Key hyperparameters:

| Parameter                       | Value      |
| ------------------------------- | ---------- |
| Batch size                      | 256        |
| Max steps                       | 50,000     |
| Learning rate                   | 5e-5       |
| Warmup steps                    | 5,000      |
| Max sequence length             | 256        |
| Weight decay                    | 0.01       |
| MLM mask probability            | 15%        |
| Contrastive loss weight (alpha) | 0.7        |
| Temperature                     | 0.07       |
| Contrastive mode                | `info_nce` |

Models are saved to `saved_models/<run_name>/`. Experiment tracking is via [Weights & Biases](https://wandb.ai).

#### Contrastive Loss Modes

The `--contra_mode` flag selects the contrastive loss function:

- **`info_nce`** (default): Standard InfoNCE with diagonal positives only. Each code sample is paired with its single augmentation; all other batch items are negatives.
- **`supcon`**: Supervised Contrastive loss ([Khosla et al., 2020](https://arxiv.org/abs/2004.11362)). Uses a `function_id` (hash of the original code) to identify all augmentations of the same function within a batch as positives. Code and augmented embeddings are concatenated into a single pool of size `2B`, and a positive mask marks all pairs sharing the same `function_id`.

```sh
# Standard InfoNCE (default)
./run_pretrain.sh

# SupCon multi-positive
python -m modeling.pretrain --contra_mode supcon ...
```

SupCon benefits from larger per-GPU batch sizes since it needs multiple augmentations of the same function to co-occur in a batch for the multi-positive signal to activate.

### Downstream Evaluation

We evaluate on 7 downstream tasks from the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) benchmark:

| Task                | Dataset                               | Metric   | Language          |
| ------------------- | ------------------------------------- | -------- | ----------------- |
| Clone Detection     | POJ-104                               | MAP@R    | C/C++             |
| Clone Detection     | CodeNet (Java250, Python800, C++1400) | MAP@R    | Java, Python, C++ |
| Clone Detection     | BigCloneBench                         | F1       | Java              |
| Defect Detection    | Devign                                | Accuracy | C                 |
| Code Classification | POJ-104                               | Accuracy | C/C++             |
| Code Classification | CodeNet (Java250, Python800, C++1400) | Accuracy | Java, Python, C++ |
| Code Translation    | CodeXGLUE                             | BLEU     | Java, C#          |

Each task has its own directory under `downstream/` with a `run.sh` script:

```sh
cd downstream/Clone-detection-POJ-104
./run.sh <pretrained_model_path> <output_dir>
```

To evaluate robustness, use the augmented test scripts:

```sh
./run_aug_test.sh <pretrained_model_path> <output_dir>
```

### Visualization

Generate t-SNE visualizations of code embeddings across models:

```sh
uv run plot/visualize.py --input_test_file dataset/aug_test.jsonl --output_file clusters.png
```

## Invariant Code Transformations

InvPT uses six semantic-preserving transformation operators:

| Operator  | Description                              | Python | Java | C/C++ |
| --------- | ---------------------------------------- | ------ | ---- | ----- |
| **VarRe** | Rename local variables to random strings | Yes    | Yes  | Yes   |
| **F2W**   | Convert for-loop to while-loop           | No     | Yes  | Yes   |
| **W2F**   | Convert while-loop to for-loop           | No     | Yes  | Yes   |
| **PP2AA** | Convert `x++` to `x += 1`                | No     | Yes  | Yes   |
| **AA2EA** | Convert `x += 1` to `x = x + 1`          | Yes    | Yes  | Yes   |
| **RevIf** | Negate condition, swap if/else branches  | Yes    | Yes  | Yes   |

Transformations are implemented at the AST level:

- **Python**: Uses the `ast` module (`ast.NodeTransformer` subclasses) in `python_transform/src/`.
- **Java**: Uses [SPAT](https://doi.org/10.1016/j.jss.2022.111304) (bundled as `SPAT-linux.jar`) in `java_transform/`.
- **C/C++**: Uses `libclang` for AST parsing in `cpp_transforms/transformations/`.
