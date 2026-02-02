import json
import os
from dataclasses import asdict
from functools import partial
from multiprocessing import cpu_count

import fire
from pathos.multiprocessing import ProcessingPool as Pool
from returns.maybe import Nothing, Some
from tqdm import tqdm

from modeling.dataloader import AugType, CodeSearchNetExample
from python_transform.transform import (
    TRANSFORMATION_MAP,
    transform_csn,
)


def load_csn(line: str) -> CodeSearchNetExample:
    return CodeSearchNetExample(**json.loads(line))


def add_aug_type(aug_type: AugType, csn: CodeSearchNetExample) -> CodeSearchNetExample:
    csn.aug_type = aug_type
    return csn


def main(input_file_path: str, output_file_path: str):
    assert os.path.exists(input_file_path) and os.path.isfile(input_file_path), (
        "Invalid input file path"
    )

    # read in the jsonl file
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    num_cpus = cpu_count()

    with Pool(num_cpus) as pool:
        csn_examples = pool.map(load_csn, lines)

    results = []
    pbar = tqdm(TRANSFORMATION_MAP.keys())
    for aug_type in pbar:
        pbar.set_description(aug_type.value)
        with Pool(num_cpus) as pool:
            csn_examples = pool.map(partial(add_aug_type, aug_type), csn_examples)
            transformed_data = pool.map(transform_csn, csn_examples)
            results.extend(transformed_data)

    with open(output_file_path, "w") as f:
        for transformed_csn in results:
            match transformed_csn:
                case Some(csn):
                    f.write(json.dumps(asdict(csn)) + "\n")
                case Nothing:
                    pass

    print("\nFinished Transformed!\n\n")


if __name__ == "__main__":
    fire.Fire(main)
