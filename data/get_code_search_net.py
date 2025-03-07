"""download CodeSearchNet dataset from huggingface and process to our format"""

from datasets import load_dataset
from modeling.dataloader import CodeSearchNetExample
from dataclasses import asdict
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import json
from tqdm import tqdm


def convert(example: dict):
    return asdict(
        CodeSearchNetExample(
            repo=example["repository_name"],
            func_name=example["func_name"],
            language=example["language"],
            code=example["func_code_string"],
            docstring=example["func_documentation_string"],
        )
    )


def save_dataset(dataset, path):
    with open(path, "w") as f:
        for example in tqdm(dataset):
            # ! d = orjson.dumps(convert(example), option=orjson.OPT_APPEND_NEWLINE) causes incomplete result
            d = json.dumps(convert(example)) + "\n"
            f.write(d)


def main(nproc: int = cpu_count()):
    dataset = load_dataset(
        "code-search-net/code_search_net",
        split="train",
        trust_remote_code=True,
        num_proc=nproc,
    )

    print("processing Python")
    py_dataset = dataset.filter(lambda x: x["language"] == "python")
    save_dataset(py_dataset, "raw_csn_py.jsonl")

    print("processing Java")
    java_dataset = dataset.filter(lambda x: x["language"] == "java")
    save_dataset(java_dataset, "raw_csn_java.jsonl")


if __name__ == "__main__":
    main()
