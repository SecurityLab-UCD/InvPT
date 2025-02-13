"""preprocess CodeNet datasets to POJ-104 format JSONL"""

import fire
import os
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm

from download import get_fullname


@dataclass
class CodeNetProgram:
    label: str  # problem id
    index: str  # unique id
    code: str  # code content


def main(subset: str, workdir: str = "./raw"):
    assert subset in ["C++1000", "C++1400", "Java250", "Python800"]

    fullname = get_fullname(subset)
    dataset_dir = os.path.join(workdir, fullname)
    logging.info(f"Processing {dataset_dir}")

    logging.info("Loading programs")
    all_programs: list[CodeNetProgram] = []
    for label, problem in enumerate(tqdm(os.listdir(dataset_dir))):
        problem_dir = os.path.join(dataset_dir, problem)
        if not os.path.isdir(problem_dir):
            continue

        for index, program in enumerate(os.listdir(problem_dir)):
            f_path = os.path.join(problem_dir, program)
            with open(f_path, "r") as f:
                code = f.read()
                all_programs.append(
                    CodeNetProgram(
                        label=str(label),
                        index=str(index),
                        code=code,
                    )
                )

    logging.info("Writing to JSONL")
    jsonl_path = f"{subset}.jsonl"
    with open(jsonl_path, "w") as f:
        for p in all_programs:
            f.write(json.dumps(p.__dict__) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
