import sys

sys.path.append("../../../")
sys.path.append("../../../python_transform")
from python_transform.get_transformed_codes import get_transformed_codes
from returns.maybe import Maybe, Nothing, Some
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import chain
from typing import Dict, Any
import fire
import json
import os

JSON_ENCODING = "latin-1"


def get_augmented_jsons(j: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Given a JSON record with keys "code", "label", and "index",
    return a list of augmented JSON records, replacing the original "code" with the transformed value
    Returns an empty list if no transformations are possible.
    """
    code, label, index = j["code"], j["label"], j["index"]
    transformed_codes = get_transformed_codes(code)
    augmented_jsons = [
        {"code": transformed_code, "label": label, "index": index}
        for transformed_code in transformed_codes
    ]
    return augmented_jsons


def add_augmented_data_in_jsonl(test_jsonl: str) -> None:
    """
    Given a path to test.jsonl file,
    Add augmented samples by transforming each test sample with RandomLocalVarName, ReverseIfElse, and addAss2EqualAss
    """
    initial_size: int
    num_cpus = cpu_count()

    with open(test_jsonl, "r", encoding=JSON_ENCODING) as f:
        all_test_json = [json.loads(json_line) for json_line in f]
        initial_size = len(all_test_json)

    with Pool(num_cpus) as pool:
        all_augmented_json = list(
            chain.from_iterable(pool.map(get_augmented_jsons, all_test_json))
        )

    with open(test_jsonl, "a") as f:
        for augmented_json in all_augmented_json:
            f.write(json.dumps(augmented_json) + "\n")

    print(
        f"Initial test.jsonl ({test_jsonl}) size: {initial_size}; added: {len(all_augmented_json)}."
    )


def validate_jsonl_path(path: str):
    assert os.path.isfile(path), f"The path {path} does not refer to a file."
    assert path.endswith(".jsonl"), f"The file {path} is not a JSONL file."


def main(test_jsonl_file):
    validate_jsonl_path(test_jsonl_file)
    add_augmented_data_in_jsonl(test_jsonl_file)
    print(f"Successfully added transformed data to {test_jsonl_file}")


if __name__ == "__main__":
    fire.Fire(main)
