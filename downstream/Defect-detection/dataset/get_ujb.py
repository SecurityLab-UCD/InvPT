# Get UJB dataset in test train eval splits.

import fire
import os
from datasets import load_dataset
from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pprint
import json


# UJB defect detection huggingface path from their GitHub repository
UJB_PATH = "ZHENGRAN/code_ujb_defectdetection"


@dataclass
class UJBExample:
    """An entry in the UJB dataset"""
    bug_id: str
    task_id: str
    function_signature: str
    prompt_chat: str
    code: str
    defective: str
    project: str
    prompt_complete: str


def main(output_dir: Path = Path("ujb")):
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    with open(output_dir / Path("train.jsonl"), "w") as f:
        for entry in load_dataset(UJB_PATH, split="train"):
            example = UJBExample(**entry)
            json.dump(
                {
                    **asdict(example),
                    "func": example.code,
                    "idx": idx,
                    "target": 1 if example.defective else 0,
                },
                f,
            )
            f.write("\n")
            idx += 1


if __name__ == "__main__":
    fire.Fire(main)
