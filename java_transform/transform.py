import json
import os
from java_transform import TRANSFORMATION_MAP
from modeling.dataloader import CodeSearchNetExample, AugType
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from tqdm import tqdm
import logging


def process(
    augtype: AugType,
    dataset: list[CodeSearchNetExample],
) -> list[CodeSearchNetExample]:
    logging.info("Preparing data for transformation")
    id_map = {i: entry for i, entry in enumerate(dataset)}
    programs = [(i, entry.code) for i, entry in id_map.items()]

    logging.info("Calling SPAT for transformation")
    transformed = TRANSFORMATION_MAP[augtype](programs)

    succeed = []
    for i, code in transformed:
        entry = id_map[i]
        entry.transformed = code
        entry.aug_type = augtype
        succeed.append(entry)
    return succeed


def main(
    augtype: AugType,
    input_file_path: str,
    output_file_path: str,
    num_cpus: int,
):
    print(f"-------- Selected Transforming Method: {augtype} -------- ")

    assert os.path.exists(input_file_path) and os.path.isfile(
        input_file_path
    ), "Invalid input file path"

    # read in the jsonl file
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    with Pool(num_cpus) as pool:
        csn: list[CodeSearchNetExample] = pool.map(
            lambda line: CodeSearchNetExample(**json.loads(line)), lines
        )

    transformed = process(augtype, csn)
    print(len(transformed))

    print("Writing results")
    with open(output_file_path, "w") as f:
        for entry in tqdm(transformed):
            f.write(json.dumps(entry.__dict__) + "\n")

    print("\nFinished Transformed!\n\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # Parsing Arguments
    arg_parser.add_argument(
        "-t",
        "--augtype",
        help="The id of transformation method",
        type=str,
        choices=[k.value for k in TRANSFORMATION_MAP.keys()],
    )
    arg_parser.add_argument(
        "-i",
        "--input_path",
        help="Path to jsonl file for transformation",
        type=str,
    )
    arg_parser.add_argument(
        "-o",
        "--output_path",
        help="The target jsonl file where the transformed code are located",
        type=str,
    )
    arg_parser.add_argument(
        "-n",
        "--num_cpus",
        help="The number of CPU cores to use for parallel processing",
        type=int,
        default=cpu_count(),
    )

    args = arg_parser.parse_args()
    main(AugType(args.augtype), args.input_path, args.output_path, args.num_cpus)
