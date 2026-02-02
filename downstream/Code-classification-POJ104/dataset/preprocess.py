"""preprocess CodeNet datasets to POJ-104 format JSONL"""

import fire
import os
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
import random


JSON_ENCODING = "utf-8"


def files(path):
    g = os.walk(path)
    file = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file.append(os.path.join(path, file_name))
    return file


@dataclass
class CodeNetProgram:
    label: int  # problem id
    index: str  # unique id
    code: str  # code content


def get_problem_ids(all_programs: list[CodeNetProgram]) -> list[int]:
    return list(set(p.label for p in all_programs))


def split_programs(
    all_programs: list[CodeNetProgram], num_of_classes: int = 104
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


def main(seed: int = 0):

    logging.info("Loading programs")
    all_programs: list[CodeNetProgram] = []
    index = 0
    for i in tqdm(range(1, 105), total=104):
        items = files("ProgramData/{}".format(i))
        for item in items:
            js = {}
            js["label"] = item.split("/")[1]
            js["index"] = str(index)
            js["code"] = open(item, encoding="latin-1").read()
            all_programs.append(
                CodeNetProgram(
                    label=int(js["label"]),
                    index=str(js["index"]),
                    code=js["code"],
                )
            )
            index += 1

    random.seed(seed)
    train_programs, valid_programs, test_programs = split_programs(all_programs)

    logging.info("Writing to JSONL")
    with open("train.jsonl", "w") as f:
        for p in train_programs:
            f.write(json.dumps(p.__dict__) + "\n")
    with open("valid.jsonl", "w") as f:
        for p in valid_programs:
            f.write(json.dumps(p.__dict__) + "\n")
    with open("test.jsonl", "w") as f:
        for p in test_programs:
            f.write(json.dumps(p.__dict__) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
