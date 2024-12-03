from dataclasses import dataclass
from enum import Enum
import fire
import json
import os
from multiprocessing import Pool

from torch.utils.data import DataLoader, IterableDataset


@dataclass
class CodeSearchNetExample:
    repo: str
    func_name: str
    original_string: str
    code: str
    language: str
    docstring: str


class CodeSearchNetDataset(IterableDataset):
    def __init__(self, content: str):
        self.lines = content.split("\n")

    def __iter__(self):
        for line in self.lines:
            example = json.loads(line)
            yield CodeSearchNetExample(
                repo=example["repo"],
                func_name=example["func_name"],
                original_string=example["original_string"],
                code=example["code"],
                language=example["language"],
                docstring=example["docstring"],
            )


def main(dataset_path: str = "data/codesearchnet.jsonl"):
    with open(dataset_path, "r") as f:
        dataset_content = f.read()

    dataset = CodeSearchNetDataset(dataset_content)

    d = iter(dataset)
    print(next(d))


if __name__ == "__main__":
    fire.Fire(main)
