# AGENTS.md

## Project Overview

**InvPT** (Invariant Pre-training) is a research project for robust code representation learning.
It continues pre-training encoder-based code models (GraphCodeBERT, ContraBERT) using
semantic-preserving code transformations combined with contrastive learning and curriculum learning.
The goal is to improve both downstream task performance and robustness against syntactically
different but semantically equivalent programs (invariant programs).

Paper: *Invariant Pre-training for Robust Code Representation Learning* (ICSE 2026)

## Repository Structure

```
InvPT/
├── modeling/              # Core training pipeline
│   ├── pretrain.py        # Entry point for pre-training
│   ├── config.py          # PretrainConfig dataclass, YAML config loading
│   ├── model.py           # ContrastiveTrainer (MLM + InfoNCE loss)
│   ├── dataloader.py      # Data collation, AugType enum, CodeSearchNetExample
│   └── common.py          # Device setup, seed utilities
├── python_transform/      # Python AST-based code transformations
│   ├── transform.py       # Transformation orchestrator
│   ├── augment_pretrain.py# Batch augmentation for pre-training data
│   ├── src/               # Individual transformers (NodeTransformer subclasses)
│   └── tests/             # Unit tests for transformations
├── java_transform/        # Java code transformations via SPAT
│   ├── transform.py       # Transformation orchestrator
│   ├── augment_pretrain.py# Batch augmentation for pre-training data
│   ├── utils.py           # SPAT JAR interface
│   └── SPAT-linux.jar     # External Java transformation tool
├── cpp_transforms/        # C/C++ transformations via libclang
│   ├── transform.py       # Transformation orchestrator
│   └── transformations/   # Individual transformation modules
├── data/                  # Pre-training dataset scripts
│   └── get_code_search_net.py  # Downloads CodeSearchNet from HuggingFace
├── downstream/            # Fine-tuning & evaluation (7 tasks from CodeXGLUE)
│   ├── Clone-detection-POJ-104/
│   ├── Clone-detection-CodeNet/
│   ├── Clone-detection-BigCloneBench/
│   ├── Code-classification-POJ104/
│   ├── Code-classification-CodeNet/
│   ├── Defect-detection/
│   └── Code-translation/
├── plot/                  # t-SNE visualization of embeddings
│   └── visualize.py
├── experiments/            # YAML experiment configurations
│   ├── base.yaml          # Base supcon config (matches original run_pretrain.sh)
│   └── grouped_example.yaml # Grouped contrastive mode example
├── saved_models/          # Pre-trained model checkpoints
├── run_pretrain.sh        # Pre-training launch script
├── clang.sh               # LLVM 14 installation script
├── pyproject.toml         # Dependencies (managed by uv)
└── .envrc                 # Environment variables (direnv)
```

## Key Concepts

- **Invariant programs**: Semantically equivalent code with different syntax (variable names, loop structures, branching). These occur naturally in real codebases.
- **Training loss**: `L = L_MLM(X) + L_MLM(X_inv) + alpha * L_InfoNCE(X, X_inv)` where alpha=0.7.
- **Curriculum learning**: Self-contrast (easy, same code with different MLM masks) and invariant-contrast (hard, transformed code) are trained simultaneously with low learning rate.
- **PL-only**: Unlike prior work (CodeBERT, ContraBERT), InvPT removes natural language docstrings during pre-training.
- **No MoCo**: Uses a single shared encoder for original and transformed code, unlike ContraBERT which uses momentum contrast.
- **Contrastive modes**: `info_nce` (diagonal positives), `supcon` (multi-positive by function_id mask), `grouped` (grouped multi-key contrast with explicit aug grouping via `--max_num_augs`).

## Transformation Operators

| Operator | Python | Java | C/C++ | Description |
|----------|--------|------|-------|-------------|
| VarRe    | yes    | yes  | yes   | Rename local variables to random strings |
| F2W      | no     | yes  | yes   | Convert for-loop to while-loop |
| W2F      | no     | yes  | yes   | Convert while-loop to for-loop |
| PP2AA    | no     | yes  | yes   | Convert `x++` to `x += 1` |
| AA2EA    | yes    | yes  | yes   | Convert `x += 1` to `x = x + 1` |
| RevIf    | yes    | yes  | yes   | Negate condition, swap if/else branches |

## Development Setup

```sh
uv sync                   # Install Python dependencies
source .envrc             # Load environment variables
./clang.sh                # Install LLVM 14 (for C/C++ transforms)
```

Requires: Python 3.11+, JDK 11+ (for Java transforms), LLVM 14 (for C/C++ transforms).

## Running Tests

```sh
pytest python_transform/tests/
```

## Pre-training Pipeline

1. Download data: `cd data && python get_code_search_net.py`
2. Augment Python: `python python_transform/augment_pretrain.py -i data/raw_csn_py.jsonl -o data/aug_csn_py.jsonl`
3. Augment Java: `python java_transform/augment_pretrain.py -i data/raw_csn_java.jsonl -o data/aug_csn_java.jsonl`
4. Combine: `cat data/raw_csn.jsonl data/aug_csn_py.jsonl data/aug_csn_java.jsonl > data/csn.jsonl`
5. Train: `./run_pretrain.sh` or `python -m modeling.pretrain --config experiments/base.yaml`

Pre-training runs for 50k steps on 4 GPUs with batch size 256, learning rate 5e-5, and 5000 warmup steps.

### Experiment Configuration

Experiments are configured via YAML files in `experiments/`. The `--config` flag loads a YAML config,
and any additional CLI args override the YAML values (precedence: dataclass defaults < YAML < CLI).

```sh
# Full YAML config
python -m modeling.pretrain --config experiments/base.yaml

# YAML config with CLI overrides for quick experiments
python -m modeling.pretrain --config experiments/base.yaml --seed 42 --run_name "seed42-test"

# Pure CLI (backward compatible, no config file)
python -m modeling.pretrain --batch_size=64 --num_epochs=3 --model_name="./saved_models/ContraBERT_G" ...
```

To create a new experiment, copy an existing YAML file in `experiments/` and modify the parameters.

## Downstream Evaluation

Each task in `downstream/` has a `run.sh` script:

```sh
cd downstream/Clone-detection-POJ-104
./run.sh <pretrained_model_path> <output_dir>
```

Metrics: MAP@R for clone detection, accuracy for defect detection and code classification.

## Code Conventions

- Type annotations are used throughout; checked with mypy (strict mode, see `mypy.ini`).
- Functional error handling via `returns` library (`Maybe`, `Some`, `Nothing`).
- Parallel processing via `pathos` (serializable lambdas).
- CLI interfaces use `argparse` or `fire`. Pre-training uses YAML configs via `modeling/config.py`.
- Experiment tracking via Weights & Biases (`wandb`).

## Development Guidelines

### Developing

- Before start working, refresh your knowledge from contents in `.agents` first.
- Always update `README.md` and `CLAUDE.md` when you introduce new features or libraries.
- Always write unit tests for integration testing and functional testing of new features.
- Always test your code after your implementation.
- Use `pre-commit install --install-hooks` (and optionally `--hook-type pre-push`) to enable local git hooks.
- You should not commit anything and create pull request, let human do them. However, please suggest commit messages after implementation.

1. Use `.agents/sandbox/` for throwaway exploration that will not be committed.
2. Use `.agents/notes/` for longer-term notes that may be useful later.
Always write down your plans and reasoning for future reference when encountering major tasks,
like adding a feature.
3. Use `.agents/accomplished/` for recording completed tasks and the summary of what we did,
this may be useful for future reference.


## Common Tasks

- **Add a new transformation operator**: Implement it in the relevant `*_transform/` directory. For Python, subclass `ast.NodeTransformer`; for Java, use a SPAT rule ID; for C/C++, use libclang AST traversal. Register it in the corresponding `TRANSFORMATION_MAP`.
- **Change the contrastive loss**: Modify `ContrastiveTrainer.compute_loss()` in `modeling/model.py`. Alternative loss functions (e.g., Barlow Twins) are already stubbed there.
- **Add a new downstream task**: Follow the CodeXGLUE structure in `downstream/`. Each task needs a `code/` directory with training scripts, a `dataset/` directory, and a `run.sh` entry point.
