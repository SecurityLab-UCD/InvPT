import os
import json
import fire
from multiprocessing import cpu_count
from java_transform.transform import TRANSFORMATION_MAP, process
from modeling.dataloader import CodeSearchNetExample
from pathos.multiprocessing import ProcessingPool as Pool


def main(input_file_path: str, output_file_path: str):
    nproc = cpu_count()

    assert os.path.exists(input_file_path) and os.path.isfile(
        input_file_path
    ), "Invalid input file path"
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    with Pool(nproc) as pool:
        csn: list[CodeSearchNetExample] = pool.map(
            lambda l: CodeSearchNetExample(**json.loads(l)), lines
        )

    transformed = []

    for augtype in TRANSFORMATION_MAP.keys():
        print(f"-------- Selected Transforming Method: {augtype} -------- ")
        transformed.extend(process(augtype, csn))

    with open(output_file_path, "w") as f:
        for entry in transformed:
            f.write(json.dumps(entry.__dict__) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
