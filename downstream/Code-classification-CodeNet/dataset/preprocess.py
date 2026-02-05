"""preprocess CodeNet datasets to POJ-104 format JSONL"""

import typer
import os
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
import random

from download import get_fullname

JSON_ENCODING = "utf-8"


@dataclass
class CodeNetProgram:
    label: int  # problem id
    index: str  # unique id
    code: str  # code content


def get_problem_ids(all_programs: list[CodeNetProgram]) -> list[int]:
    return list(set(p.label for p in all_programs))


def split_programs(
    all_programs: list[CodeNetProgram],
    num_of_classes: int = 104
) -> tuple[list[CodeNetProgram], list[CodeNetProgram], list[CodeNetProgram]]:
    """split programs into train, valid, test sets by 50%, 25%, 25%"""
    # split the dataset by problems, not by programs
    # train, valid, test = 0.5, 0.25, 0.25
    pids = list(set(p.label for p in all_programs))
    sampled_pids = random.sample(pids, num_of_classes)
    sampled_pids_maps = {}
    for idx, pid in enumerate(sampled_pids):
        sampled_pids_maps[pid] = int(idx)

    train_programs = []
    valid_programs = []
    test_programs = []

    index = 0
    random.shuffle(all_programs)
    for p in all_programs:
        if p.label in sampled_pids_maps:
            p.label = sampled_pids_maps[p.label]
            if index % 4 == 0:
                valid_programs.append(p)
            elif index % 4 == 1:
                test_programs.append(p)
            else:
                train_programs.append(p)
        index += 1

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
                        label=int(label),
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
    typer.run(main)
