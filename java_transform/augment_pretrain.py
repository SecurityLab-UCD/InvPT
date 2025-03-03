from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm
from utils import jsonl_to_df, decompose, id_to_name, DIR_PATH
import numpy as np
import os
import json
import fire
import pandas as pd
import subprocess


def postprocess(
    original: pd.DataFrame,
    transformed_path: Path,
    output_path: Path,
    rule_id: int,
):
    """Processes SPAT output and append to `output_path`.

    Args:
    original -- the original dataset (code: str)
    transformed_path -- the path of the files transformed by SPAT
    output_path -- the path to append the SPAT output to
    rule_id -- the ID of the SPAT transformation rule
    """
    with open(output_path, "a") as output_file:
        for file in tqdm(os.listdir(transformed_path), desc="Processing SPAT output"):
            aug_from = int(file.lstrip("n").rstrip(".java"))
            with open(transformed_path / file) as transformed_file:
                transformed = transformed_file.read()
            entry = {
                **original.loc[aug_from],
                "aug_type": id_to_name[rule_id],
                "transformed": transformed,
            }
            output_file.write(f"{json.dumps(entry)}\n")


def main(
    input_path: str,
    output_path: str,
    spat_jar: str = str(DIR_PATH / "SPAT-linux.jar"),
    spat_lib: str = "/usr/lib/jvm/java-18-openjdk-amd64/lib",
    rules: list[int] = [0, 1, 2, 3, 6, 7],
):
    """Augment pretrain dataset

    Args:
    input_path: Path to the input jsonl (code: str)
    output_path: Path to save the augmented jsonl (code: str, transformed: str,
        aug_type: str)
    spat_jar: Path to the SPAT jar file used for augmentation
    spat_lib: Path to the Java library used by SPAT (see SPAT documentation)
    rules: List of SPAT rule IDs.
    """
    original = jsonl_to_df(input_path)
    print("Processing original dataset. This might take a minute...")
    original["transformed"] = original["code"]
    original["aug_type"] = pd.Series(
        np.full(original.shape[0], "None"), index=original.index
    )
    with open(output_path, "w") as f:
        f.write(original.to_json(orient="records", lines=True, force_ascii=False))
    print("Augmenting original dataset")
    with TemporaryDirectory() as original_dir:
        decompose(original, Path(original_dir))
        for rule in rules:
            print(f"Augmenting dataset with rule {rule}...")
            with TemporaryDirectory() as transformed_dir:
                subprocess.run(
                    [
                        "java",
                        "-jar",
                        spat_jar,
                        str(rule),
                        original_dir,
                        transformed_dir,
                        spat_lib,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                postprocess(original, Path(transformed_dir), Path(output_path), rule)


if __name__ == "__main__":
    fire.Fire(main)
