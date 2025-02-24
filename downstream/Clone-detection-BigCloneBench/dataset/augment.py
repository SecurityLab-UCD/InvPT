import fire
import pandas as pd
import subprocess
import os
import logging
from tqdm import tqdm
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_PATH = Path("./data.jsonl")
TMP_PATH = Path("./tmp.jsonl")
AUGDATA_PATH = Path("./data.jsonl")
TEST_PATH = Path("./test.txt")
AUGTEST_PATH = Path("./aug_test.txt")


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
    logger.info(f"Preprocessing {DATA_PATH} for augmentation...")
    df = jsonl_to_df(DATA_PATH)
    df = df.rename({"func": "code", "idx": "index"}, axis="columns")
    with open(TMP_PATH, "w") as f:
        f.write(df.to_json(orient="records", lines=True, force_ascii=False))

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
    with open(TEST_PATH, "r") as file:
        num_tests = sum(1 for _ in file)
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
