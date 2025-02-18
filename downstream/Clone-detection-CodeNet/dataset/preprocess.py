"""preprocess CodeNet datasets to POJ-104 format JSONL"""

import fire
import os
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
import random

from download import get_fullname


@dataclass
class CodeNetProgram:
    label: str  # problem id
    index: str  # unique id
    code: str  # code content


def get_problem_ids(all_programs: list[CodeNetProgram]) -> list[str]:
    return list(set(p.label for p in all_programs))


def split_programs(
    all_programs: list[CodeNetProgram],
) -> tuple[list[CodeNetProgram], list[CodeNetProgram], list[CodeNetProgram]]:
    """split programs into train, valid, test sets by 50%, 25%, 25%"""
    # split the dataset by problems, not by programs
    # train, valid, test = 0.5, 0.25, 0.25
    pids = list(set(p.label for p in all_programs))
    random.shuffle(pids)
    split1 = len(pids) // 2
    split2 = len(pids) // 4 + split1
    train_pids = pids[:split1]
    valid_pids = pids[split1:split2]
    test_pids = pids[split2:]

    train_programs = []
    valid_programs = []
    test_programs = []
    for p in all_programs:
        if p.label in train_pids:
            train_programs.append(p)
        elif p.label in valid_pids:
            valid_programs.append(p)
        elif p.label in test_pids:
            test_programs.append(p)
        else:
            raise ValueError("Invalid problem id")

    return train_programs, valid_programs, test_programs


def main(subset: str, workdir: str = "./raw", seed: int = 0):
    assert subset in ["C++1000", "C++1400", "Java250", "Python800"]

    fullname = get_fullname(subset)
    dataset_dir = os.path.join(workdir, fullname)
    logging.info(f"Processing {dataset_dir}")

    logging.info("Loading programs")
    all_programs: list[CodeNetProgram] = []
    index = 0
    for label, problem in enumerate(tqdm(os.listdir(dataset_dir))):
        problem_dir = os.path.join(dataset_dir, problem)
        if not os.path.isdir(problem_dir):
            continue

        for program in os.listdir(problem_dir):
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
                index += 1

    random.seed(seed)
    train_programs, valid_programs, test_programs = split_programs(all_programs)

    logging.info("Writing to JSONL")
    os.makedirs(subset, exist_ok=True)
    with open(f"{subset}/train.jsonl", "w") as f:
        for p in train_programs:
            f.write(json.dumps(p.__dict__) + "\n")
    with open(f"{subset}/valid.jsonl", "w") as f:
        for p in valid_programs:
            f.write(json.dumps(p.__dict__) + "\n")
    with open(f"{subset}/test.jsonl", "w") as f:
        for p in test_programs:
            f.write(json.dumps(p.__dict__) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
