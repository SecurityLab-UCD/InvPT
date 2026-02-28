from cpp_transforms.transform import apply_code_transformation, augment_accumulatively
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from dataclasses import asdict
from functools import partial
import typer
import json
import os
from modeling.dataloader import AugType
from preprocess import CodeNetProgram

JSON_ENCODING = "utf-8"


def resolve_operator(operator_key: str) -> AugType | None:
    key = operator_key.strip().lower()
    if not key:
        return None
    for aug_type in AugType:
        if aug_type.name.lower() == key:
            return aug_type
    available = ", ".join(aug_type.name.lower() for aug_type in AugType)
    raise ValueError(f"Unsupported operator: {operator_key}. Available: {available}")


def augment_single(aug_type: AugType, program: CodeNetProgram) -> CodeNetProgram:
    code = apply_code_transformation(aug_type, program.code).value_or(program.code)
    return CodeNetProgram(label=program.label, index=program.index, code=code)


def validate_jsonl_path(path: str):
    assert os.path.isfile(path), f"The path {path} does not refer to a file."
    assert path.endswith(".jsonl"), f"The file {path} is not a JSONL file."


def main(
    input_file_path: str,
    output_file_path: str,
    nproc: int = cpu_count(),
    operator_key: str = "",
):
    validate_jsonl_path(input_file_path)
    operator = resolve_operator(operator_key)

    with open(input_file_path, "r", encoding=JSON_ENCODING) as f:
        all_test_json = [CodeNetProgram(**json.loads(json_line)) for json_line in f]

    with Pool(nproc) as pool:
        if operator is None:
            augmented_dataset = pool.map(augment_accumulatively, all_test_json)
        else:
            augmented_dataset = pool.map(
                partial(augment_single, operator), all_test_json
            )
        augmented_jsonl = pool.map(lambda j: json.dumps(asdict(j)), augmented_dataset)

    with open(output_file_path, "w", encoding=JSON_ENCODING) as f:
        for aj in augmented_jsonl:
            f.write(aj + "\n")

    if operator is None:
        print(f"Successfully added transformed data to {output_file_path}")
    else:
        print(f"Successfully added {operator.name.lower()} data to {output_file_path}")


if __name__ == "__main__":
    typer.run(main)
