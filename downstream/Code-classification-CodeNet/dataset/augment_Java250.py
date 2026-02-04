import typer
import json
from dataclasses import asdict
from multiprocessing import cpu_count
from preprocess import CodeNetProgram, JSON_ENCODING
from pathos.multiprocessing import ProcessingPool as Pool
import logging
from java_transform import augment_accumulatively


def process(dataset: list[CodeNetProgram]) -> list[CodeNetProgram]:
    dataset = dataset.copy()
    logging.info("Preparing data for transformation")
    id_map = {i: entry for i, entry in enumerate(dataset)}
    programs = [(i, entry.code) for i, entry in id_map.items()]

    logging.info("Applying SPAT for transformation")
    transformed = augment_accumulatively(programs)

    succeed = []
    for i, code in transformed:
        entry = id_map[i]
        entry.code = code
        succeed.append(entry)

    return succeed


def main(
    input_file_path: str,
    output_file_path: str,
    nproc: int = cpu_count(),
):

    with open(input_file_path, "r", encoding=JSON_ENCODING) as f:
        all_test_json = [CodeNetProgram(**json.loads(json_line)) for json_line in f]

    augmented_dataset = process(all_test_json)

    with Pool(nproc) as pool:
        augmented_jsonl = pool.map(lambda j: json.dumps(asdict(j)), augmented_dataset)

    with open(output_file_path, "w", encoding=JSON_ENCODING) as f:
        for aj in augmented_jsonl:
            f.write(aj + "\n")

    print(f"Successfully added transformed data to {output_file_path}")


if __name__ == "__main__":
    typer.run(main)
