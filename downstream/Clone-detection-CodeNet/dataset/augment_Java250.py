import typer
import json
from dataclasses import asdict
from multiprocessing import cpu_count
from preprocess import CodeNetProgram, JSON_ENCODING
import logging
from java_transform import TRANSFORMATION_MAP, augment_accumulatively
from modeling.dataloader import AugType


def resolve_operator(operator_key: str) -> AugType | None:
    key = operator_key.strip().lower()
    if not key:
        return None
    for aug_type in TRANSFORMATION_MAP:
        if aug_type.name.lower() == key:
            return aug_type
    available = ", ".join(aug_type.name.lower() for aug_type in TRANSFORMATION_MAP)
    raise ValueError(
        f"Unsupported operator for Java: {operator_key}. Available: {available}"
    )


def process(
    dataset: list[CodeNetProgram],
    operator: AugType | None = None,
) -> list[CodeNetProgram]:
    logging.info("Preparing data for transformation")
    programs = [(i, entry.code) for i, entry in enumerate(dataset)]

    if operator is None:
        logging.info("Applying SPAT cumulative transformations")
        transformed = augment_accumulatively(programs)
    else:
        logging.info("Applying SPAT transformation: %s", operator.name.lower())
        transformed = TRANSFORMATION_MAP[operator](programs)
    transformed_map = {i: code for i, code in transformed}

    result: list[CodeNetProgram] = []
    for i, entry in enumerate(dataset):
        code = transformed_map.get(i, entry.code)
        result.append(CodeNetProgram(label=entry.label, index=entry.index, code=code))
    return result


def main(
    input_file_path: str,
    output_file_path: str,
    nproc: int = cpu_count(),
    operator_key: str = "",
):
    del nproc  # SPAT runs in batch; retained for CLI compatibility.
    operator = resolve_operator(operator_key)

    with open(input_file_path, "r", encoding=JSON_ENCODING) as f:
        all_test_json = [CodeNetProgram(**json.loads(json_line)) for json_line in f]

    augmented_dataset = process(all_test_json, operator)
    augmented_jsonl = [json.dumps(asdict(j)) for j in augmented_dataset]

    with open(output_file_path, "w", encoding=JSON_ENCODING) as f:
        for aj in augmented_jsonl:
            f.write(aj + "\n")

    if operator is None:
        print(f"Successfully added transformed data to {output_file_path}")
    else:
        print(f"Successfully added {operator.name.lower()} data to {output_file_path}")


if __name__ == "__main__":
    typer.run(main)
