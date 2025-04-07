# Get UJB dataset in test train eval splits.

import fire
from datasets import load_dataset
from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pprint
import json


# UJB defect detection huggingface path from their GitHub repository
UJB_PATH = "ZHENGRAN/code_ujb_defectdetection"

@dataclass
class UJBExample():
    """An entry in the UJB dataset"""
    bug_id: str
    task_id: str
    function_signature: str
    prompt_chat: str
    code: str
    defective: str
    project: str
    prompt_complete: str

    def get_codexglue(self):
        """Get example in codexglue format"""
        return {
            **asdict(self),
            "func": self.code,
            "idx": self.task_id,
            "target": 1 if self.defective else 0
        }

def main(output_dir: Path):
    with open(output_dir / Path("train.jsonl"), "w") as f:
        for entry in load_dataset(UJB_PATH, split='train[:70%]'):
            example = UJBExample(**entry)
            json.dump(example.get_codexglue(), f)
            f.write("\n")
    with open(output_dir / Path("test.jsonl"), "w") as f:
        for entry in load_dataset(UJB_PATH, split='train[70%:]'):
            example = UJBExample(**entry)
            json.dump(example.get_codexglue(), f)
            f.write("\n")

if __name__ == "__main__":
    fire.Fire(main)
