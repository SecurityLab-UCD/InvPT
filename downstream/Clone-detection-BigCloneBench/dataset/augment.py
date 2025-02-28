import fire
import pdb
import pandas as pd
import numpy as np
import subprocess
import os
import logging
from tqdm import tqdm
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pre-augmentation downloaded dataset
ORIGINAL_PATH = Path("./original_data.jsonl")
# Path used by BCB benchmark
DATA_PATH = Path("./data.jsonl")
# Data formatted in POJ104
FMTDATA_PATH = Path("./fmtdata.jsonl")
TMP_PATH = Path("./tmp.jsonl")
# Augmented dataset
AUGDATA_PATH = Path("./data.jsonl")
# Pre-augmentation test.txt
TEST_PATH = Path("./test.txt")
# Augmented test.txt
AUGTEST_PATH = Path("./aug_test.txt")

# From geeksforgeeks.org
class UnionFind:
    def __init__(self):
        self.Parent = {}
        self.Size = {}
    def find(self, i):
        root = self.Parent.get(i)
        if root is None:
            self.Parent[i] = i
            self.Size[i] = 1
            root = i
        if self.Parent[root] != root:
            self.Parent[i] = self.find(root)
            return self.Parent[i]
        return root
    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        if irep == jrep:
            return
        isize = self.Size[irep]
        jsize = self.Size[jrep]
        if isize < jsize:
            self.Parent[irep] = jrep
            self.Size[jrep] += self.Size[irep]
        else:
            self.Parent[jrep] = irep
            self.Size[irep] += self.Size[jrep]


def jsonl_to_df(path, chunksize=1000):
    with open(path, "r") as file:
        # Count total lines in the file
        total_lines = sum(1 for _ in file)

    with open(path, "r") as file, tqdm(
        total=total_lines, desc=f"reading {path}"
    ) as pbar:
        chunks = []
        for chunk in pd.read_json(file, lines=True, chunksize=chunksize):
            chunks.append(chunk)
            pbar.update(chunksize)
        df = pd.concat(chunks, ignore_index=True)
        logger.info("read complete!")
        return df


def main():
    """Augments BCB dataset"""
    logger.info(f"Preprocessing {ORIGINAL_PATH} for augmentation...")
    df = jsonl_to_df(ORIGINAL_PATH)
    df = df.rename({"func": "code", "idx": "index"}, axis="columns")
    print(df)
    df["label"] = pd.Series(df["index"])
    df = df.set_index("index")
    print(df)
    with open(TEST_PATH, "r") as file:
        num_tests = sum(1 for _ in file)
    union_find = UnionFind()
    with open(TEST_PATH, 'r') as in_file:
        for line in tqdm(in_file, total=num_tests, desc="Labeling dataset"):
            line = line.split()
            is_similar = line[2]
            if is_similar != "1":
                continue
            idx1 = line[0]
            idx2 = line[1]
            pdb.set_trace()
            union_find.union(idx1, idx2)
    for idx, parent in union_find.Parent.items():
        df.loc[idx, 'label'] = parent
    print(df)
    pdb.set_trace()

    with open(TMP_PATH, "w") as f:
        f.write(df.reset_index().to_json(orient="records", lines=True, force_ascii=False))
    return df
    logger.info(f"Augmenting {DATA_PATH}...")
    subprocess.run(
        [
            "python3",
            "../../../java_transform/augment.py",
            TMP_PATH,
            AUGDATA_PATH,
        ],
    )

    logger.info(f"Augmenting {TEST_PATH}...")
    df = jsonl_to_df(AUGDATA_PATH)
    df = df.set_index('aug_from')
    with open(AUGTEST_PATH, 'w') as out_file, open(TEST_PATH, 'r') as in_file:
        for line in tqdm(in_file, total=num_tests):
            line = line.split()
            idx1 = line[0]
            idx2 = line[1]
            is_similar = line[2]

            idx1_subs = df.loc[int(idx1), 'index']
            idx2_subs = df.loc[int(idx2), 'index']
            combinations = [(idx1,idx2) for idx1 in idx1_subs for idx2 in idx2_subs]
            for idx1, idx2 in combinations:
                out_file.write(f"{idx1}\t{idx2}\t{is_similar}\n")

    # Preprocess dataset for testing
    logger.info(f"Preprocessing {DATA_PATH} for testing...")
    df = df.rename({"code": "func", "index": "idx"}, axis="columns")
    with open(AUGDATA_PATH, "w") as f:
        f.write(df.to_json(orient="records", lines=True, force_ascii=False))
    os.remove(TMP_PATH)


if __name__ == "__main__":
    fire.Fire(main)
