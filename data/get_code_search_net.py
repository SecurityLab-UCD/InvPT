"""download CodeSearchNet dataset from huggingface and process to our format"""

from datasets import load_dataset
from modeling.dataloader import CodeSearchNetExample
from dataclasses import asdict
from multiprocessing import cpu_count


def convert(example: dict) -> dict:
    return asdict(
        CodeSearchNetExample(
            repo=example["repository_name"],
            func_name=example["func_name"],
            language=example["language"],
            code=example["func_code_string"],
            docstring=example["func_documentation_string"],
        )
    )


def main(nproc: int = cpu_count()):
    dataset = load_dataset(
        "code-search-net/code_search_net",
        split="train",
        trust_remote_code=True,
        num_proc=nproc,
    )

    print("converting dataset")
    converted = dataset.map(
        convert,
        remove_columns=dataset.column_names,
        num_proc=nproc,
        desc="Converting",
    )

    print("saving dataset")
    converted.to_json("raw_csn.jsonl")

    print("processing Python")
    converted.filter(lambda x: x["language"] == "python", num_proc=nproc).to_json(
        "raw_csn_py.jsonl"
    )

    print("processing Java")
    converted.filter(lambda x: x["language"] == "java", num_proc=nproc).to_json(
        "raw_csn_java.jsonl"
    )


if __name__ == "__main__":
    main()
