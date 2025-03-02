from python_transform.transform import apply_code_transformation
from modeling.dataloader import AugType
from returns.maybe import Maybe, Nothing, Some
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import chain
from typing import Dict, Any
import random
import fire
import json
import os

JSON_ENCODING = "utf-8"

def augment_accumulatively(j: Dict[str, Any]) -> None:    
    for aug_type in AugType.get_python_transformations():
        j["code"] = apply_code_transformation(aug_type, j["code"])
    return j


def augment_jsonl(test_jsonl: str) -> None:
    num_cpus = cpu_count()

    with open(test_jsonl, "r", encoding=JSON_ENCODING) as f:
        all_test_json = [json.loads(json_line) for json_line in f]
        initial_size = len(all_test_json)

    with Pool(num_cpus) as pool:
        augmented_test_jsonl = pool.map(augment_accumulatively, all_test_json)

    with open(test_jsonl, "w") as f:
        for ajs in augmented_test_jsonl:
            f.write(json.dumps(ajs) + "\n")

    print(f"Finished augmentation!")


def validate_jsonl_path(path: str):
    assert os.path.isfile(path), f"The path {path} does not refer to a file."
    assert path.endswith(".jsonl"), f"The file {path} is not a JSONL file."


def main(test_jsonl_file):
    validate_jsonl_path(test_jsonl_file)
    augment_jsonl(test_jsonl_file)
    print(f"Successfully added transformed data to {test_jsonl_file}")


if __name__ == "__main__":
    fire.Fire(main)
