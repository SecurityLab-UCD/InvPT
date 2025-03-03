import fire
import pandas as pd
import subprocess
import logging
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pre-augmentation downloaded dataset
ORIGINAL = Path("./original_data.jsonl")
# Path used by BCB benchmark
DATA = Path("./data.jsonl")
AUGSCRIPT_PATH = Path("../../../java_transform/augment_test.py")


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
    with NamedTemporaryFile() as preprocessed, NamedTemporaryFile() as augmented, open(
        DATA, "w"
    ) as data:
        logger.info(
            f"Preprocessing {ORIGINAL} to {preprocessed.name} for augmentation..."
        )
        df = jsonl_to_df(ORIGINAL)
        df = df.rename({"func": "code", "idx": "index"}, axis="columns")
        df.reset_index().to_json(
            preprocessed, orient="records", lines=True, force_ascii=False
        )

        logger.info(f"Augmenting {preprocessed.name} to {augmented.name}...")
        subprocess.run(
            [
                "python3",
                AUGSCRIPT_PATH,
                preprocessed.name,
                augmented.name,
            ],
        )

        # Preprocess dataset for testing
        logger.info(f"Postprocessing {augmented.name} for testing...")
        df = jsonl_to_df(augmented.name)
        df = df.rename({"code": "func", "aug_from_5": "idx"}, axis="columns")
        df = df[["func", "idx"]]
        df.to_json(data, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
