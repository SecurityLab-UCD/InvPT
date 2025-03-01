import json
from tqdm import tqdm
from pathlib import Path

LARGETRAIN_PATH = Path("CodeSearchNet/java/large_train.jsonl")
STRIPPED_PATH = Path("codesearchnet_java.jsonl")

found = set()

if __name__ == "__main__":
    with open(LARGETRAIN_PATH, "r") as f:
        total_lines = sum(1 for _ in f)
    with open(LARGETRAIN_PATH, "r") as large_train, \
         open(STRIPPED_PATH, "w") as stripped:
        for line in tqdm(large_train, total=total_lines, unit="line"):
            entry = json.loads(line)
            string = entry["original_string"]
            if string in found:
                continue
            found.add(string)
            stripped.write(line)
