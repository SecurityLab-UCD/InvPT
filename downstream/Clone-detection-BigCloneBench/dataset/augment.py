import fire
import pandas as pd
import subprocess
import os
import logging
from tqdm import tqdm
from tempfile import TemporaryFile
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pre-augmentation downloaded dataset
ORIGINAL_PATH = Path("./original_data.jsonl")
# Path used by BCB benchmark
DATA_PATH = Path("./data.jsonl")

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
    with TemporaryFile() as preprocessed, TemporaryFile() as augmented, open(DATA_PATH, 'w') as data:
        logger.info(f"Preprocessing {ORIGINAL_PATH} to {preprocessed} for augmentation...")
        df = jsonl_to_df(ORIGINAL_PATH)
        df = df.rename({"func": "code", "idx": "index"}, axis="columns")
        preprocessed.write(df.reset_index().to_json(orient="records", lines=True, force_ascii=False))

        logger.info(f"Augmenting {preprocessed} to {augmented}...")
        subprocess.run(
            [
                "python3",
                "../../../java_transform/augment.py",
                str(preprocessed),
                str(augmented),
            ],
        )

        # Preprocess dataset for testing
        logger.info(f"Postprocessing {augmented} for testing...")
        df = df.rename({"code": "func", "index": "idx"}, axis="columns")
        data.write(df.to_json(orient="records", lines=True, force_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
