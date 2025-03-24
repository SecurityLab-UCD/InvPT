from cpp_transforms.transform import apply_code_transformation, TRANSFORMATION_MAP
from modeling.dataloader import AugType
from returns.maybe import Maybe, Nothing, Some
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from dataclasses import asdict
import fire
import json
import os
import clang
from preprocess import CodeNetProgram


JSON_ENCODING = "utf-8"


def augment_accumulatively(j: CodeNetProgram) -> CodeNetProgram:
    code = j.code
    for aug_type in TRANSFORMATION_MAP.keys():
        code = apply_code_transformation(aug_type, code).value_or(code)
    return CodeNetProgram(label=j.label, index=j.index, code=code)


def validate_jsonl_path(path: str):
    assert os.path.isfile(path), f"The path {path} does not refer to a file."
    assert path.endswith(".jsonl"), f"The file {path} is not a JSONL file."


def main(
    input_file_path: str,
    output_file_path: str = "augmented_C++1000_test.jsonl",
    nproc: int = cpu_count(),
):
    validate_jsonl_path(input_file_path)

    with open(input_file_path, "r", encoding=JSON_ENCODING) as f:
        all_test_json = [CodeNetProgram(**json.loads(json_line)) for json_line in f]

    with Pool(nproc) as pool:
        augmented_dataset = pool.map(augment_accumulatively, all_test_json)
        augmented_jsonl = pool.map(lambda j: json.dumps(asdict(j)), augmented_dataset)

    with open(output_file_path, "w", encoding=JSON_ENCODING) as f:
        for aj in augmented_jsonl:
            f.write(aj + "\n")

    print(f"Successfully added transformed data to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
