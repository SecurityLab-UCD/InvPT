# PIA

Program-Invariant-Aware Training for Large Language Models in Code Understanding

## Usage

### Environment

#### Python

We use [`uv`](https://docs.astral.sh/uv/guides/install-python/) to manage Python environments.

```sh
uv sync
```

#### Java

We need a Java 11+ JDK for the Java augmentation.
For our experiments, we use OpenJDK21.

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

This will install the required packages for C/C++ augmentation to `/home/your_username/clang+llvm`.
Please make sure to provide this path in `.envrc`:

```sh
export LLVM=$HOME/clang+llvm
export LIBCLANG_PATH=$LLVM/lib/libclang.so
export LD_LIBRARY_PATH=$LLVM/lib:$LD_LIBRARY_PATH
```

#### **Loading Environment Variables**

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
cd data
python get_code_search_net.py
```

This script will download the CodeSearchNet dataset and convert it to jsonl format.
It will writes to the following files:

- `raw_csn.jsonl`: the entire CodeSearchNet dataset
- `raw_csn_py.jsonl`, `raw_csn_java.jsonl`: the Python and Java subset of CodeSearchNet

After conforming the dataset is downloaded, we need to transform the dataset to the format used in our paper.

```sh
python ../python_transform/augment_pretrain.py -i raw_csn_py.jsonl -o aug_csn_py.jsonl
python ../java_transform/augment_pretrain.py -i raw_csn_java.jsonl -o aug_csv_java.jsonl

mv raw_csn.jsonl csn.jsonl
cat aug_csn_py.jsonl >> csn.jsonl
cat aug_csv_java.jsonl >> csn.jsonl
```

The resulting file `csn.jsonl` will be used for pre-training.

### Pre-training RoBERTa

```bash
./run_pretrain.sh
```
