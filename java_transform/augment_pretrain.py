import os
import json
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
import typer
from java_transform import TRANSFORMATION_MAP
from java_transform.transform import process
from modeling.dataloader import CodeSearchNetExample


def main(input_file_path: str, output_file_path: str):
    """
    Augment the Java pretraining dataset using various code transformations.
    """
    nproc = cpu_count()

    assert os.path.exists(input_file_path) and os.path.isfile(input_file_path), (
        "Invalid input file path"
    )
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    with Pool(nproc) as pool:
        csn: list[CodeSearchNetExample] = pool.map(
            lambda line: CodeSearchNetExample(**json.loads(line)), lines
        )

    transformed = []

    for augtype in TRANSFORMATION_MAP.keys():
        print(f"-------- Selected Transforming Method: {augtype} -------- ")
        transformed.extend(process(augtype, csn))

    with open(output_file_path, "w") as f:
        for entry in transformed:
            f.write(json.dumps(entry.__dict__) + "\n")


if __name__ == "__main__":
    typer.run(main)
