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

The CLI entry point is `modeling/cli.py`, which provides two subcommands:

- **`run`** -- load a YAML experiment config (recommended; all parameters come from the config file to ensure full reproducibility)
- **`pretrain`** -- pass all parameters directly as CLI options

Pre-training uses PyTorch DistributedDataParallel (DDP) via `torchrun`, which ships with PyTorch itself (no extra dependencies). The same training code runs on both multi-GPU and single-GPU nodes -- the HuggingFace `Trainer` auto-detects the distributed environment set up by `torchrun` and enables or disables DDP accordingly.

#### Multi-GPU node

Use `torchrun` to spawn one process per GPU. `--nproc_per_node=gpu` automatically uses all visible GPUs:

```sh
torchrun --nproc_per_node=gpu modeling/cli.py run experiments/base.yaml
```

To select specific GPUs, set `CUDA_VISIBLE_DEVICES`:

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=gpu modeling/cli.py run experiments/base.yaml
```

There is also a convenience script that launches `torchrun` on all visible GPUs:

```sh
./run_pretrain.sh
```

#### Single-GPU node

On a single-GPU machine, run with plain `python` -- no `torchrun` needed:

```sh
python modeling/cli.py run experiments/base.yaml
```

`torchrun --nproc_per_node=1` also works if you prefer a uniform launch command across environments.

#### CLI examples

```sh
# Pass all parameters directly (without a YAML config)
python modeling/cli.py pretrain --batch-size 64 --num-epochs 3 --model-name ./saved_models/ContraBERT_G

# See all options
python modeling/cli.py run --help
python modeling/cli.py pretrain --help
```

The `batch_size` in config is the **total** batch size across all GPUs. It is automatically divided by the number of processes. For example, `batch_size: 128` on 2 GPUs gives 64 per GPU; with `gradient_accumulation_steps: 2` the effective batch size is 256.

#### Experiment Configs

YAML experiment configs live in `experiments/`. To create a new experiment, copy an existing file and modify the parameters:

```yaml
# experiments/base.yaml
dataset_path: "data/aug_csn.jsonl"
model_name: "./saved_models/ContraBERT_G"
tokenizer_name: "microsoft/graphcodebert-base"

batch_size: 64
num_epochs: 3
gradient_accumulation_steps: 4
learning_rate: 2.0e-5

seed: 0
run_name: "InvContraBERT_G-aug02-supcon"

alpha: 1.0
temperature: 0.1
max_seq_length: 512
sample_rate: 0.2

contra_mode: "supcon"
```

Key hyperparameters:

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
| Max augs per anchor (grouped)   | 6          |

Models are saved to `saved_models/<run_name>/`. Experiment tracking is via [Weights & Biases](https://wandb.ai).

#### CLI Options (`pretrain` subcommand)

The `pretrain` subcommand accepts all training parameters directly as CLI options. The `run` subcommand only takes a config file path -- edit the YAML file to change parameters.

| Option | Default | Description |
| --- | --- | --- |
| `--dataset-path` | `data/csn_jp.jsonl` | Path to the pre-training JSONL dataset |
| `--model-name` | `microsoft/codebert-base` | Pre-trained model name or path |
| `--tokenizer-name` | (same as model) | Tokenizer name; useful when model only provides weights |
| `--checkpoint` | None | Path to a checkpoint to resume weights from |
| `--batch-size` | `256` | Total batch size across all GPUs |
| `--num-epochs` | `10` | Number of training epochs |
| `--gradient-accumulation-steps` | `1` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--alpha` | `1.0` | Contrastive loss weight |
| `--temperature` | `0.07` | Contrastive loss temperature |
| `--max-seq-length` | `256` | Maximum token sequence length |
| `--sample-rate` | `1.0` | Fraction of dataset to use (for quick experiments) |
| `--seed` | `0` | Random seed |
| `--run-name` | `InvariantBERT` | W&B run name and output directory name |
| `--num-proc` | (all CPU cores) | Parallel workers for dataset preprocessing |
| `--resume / --no-resume` | `False` | Resume training from the latest checkpoint |
| `--contra-mode` | `info_nce` | Contrastive loss mode: `info_nce`, `supcon`, or `grouped` |
| `--max-num-augs` | `6` | Max augmentations per anchor group (`grouped` mode only) |

Note: dataset preprocessing uses HuggingFace Datasets multiprocessing. When running with `torchrun` (multi-GPU), `--num-proc` is automatically scaled down per-rank to avoid CPU oversubscription, and `TOKENIZERS_PARALLELISM` is disabled when using multiple workers.

#### Contrastive Loss Modes

The `--contra-mode` option selects the contrastive loss function:

- **`info_nce`** (default): Standard InfoNCE with diagonal positives only. Each code sample is paired with its single augmentation; all other batch items are negatives.
- **`supcon`**: Supervised Contrastive loss ([Khosla et al., 2020](https://arxiv.org/abs/2004.11362)). Uses a `function_id` (hash of the original code) to identify all augmentations of the same function within a batch as positives. Code and augmented embeddings are concatenated into a single pool of size `2B`, and a positive mask marks all pairs sharing the same `function_id`.
- **`grouped`**: Grouped Multi-Key Contrast. Regroups flat `(code, transformed)` rows by `function_id` at dataset load time so each batch item bundles an anchor with all of its augmentations. Positives are the anchor's own augmentations; negatives are all other anchors and their augmentations. Uses per-positive log-prob averaging with log-sum-exp stabilization. The `max_num_augs` config option (default 6) caps the number of augmentations per anchor group.

To switch contrastive modes, set the `contra_mode` field in the YAML config:

```yaml
# Standard InfoNCE (default)
contra_mode: "info_nce"

# SupCon multi-positive
contra_mode: "supcon"

# Grouped multi-key contrast
contra_mode: "grouped"
max_num_augs: 6
```

SupCon benefits from larger per-GPU batch sizes since it needs multiple augmentations of the same function to co-occur in a batch for the multi-positive signal to activate. Grouped mode guarantees all augmentations are co-located but requires more memory per batch item (each item encodes up to `max_num_augs` augmentation views); use reduced batch size with higher gradient accumulation steps.

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
