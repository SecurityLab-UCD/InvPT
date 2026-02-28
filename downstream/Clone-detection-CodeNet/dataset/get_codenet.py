from download import main as download
from preprocess import main as preprocess
import typer
import tempfile
import logging
import os
from modeling.dataloader import AugType

from augment_Python800 import main as augment_Python800
from augment_Java250 import main as augment_Java250
from augment_Cpp import main as augment_Cpp

ALL_OPERATOR_KEYS = tuple(aug_type.name.lower() for aug_type in AugType)
PYTHON_OPERATOR_KEYS = (
    AugType.LOCALVARRENAMING.name.lower(),
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT.name.lower(),
    AugType.REVERSEIFELSE.name.lower(),
)


def get_operator_keys(subset: str) -> tuple[str, ...]:
    if subset == "Python800":
        return PYTHON_OPERATOR_KEYS
    return ALL_OPERATOR_KEYS


def process(subset: str, per_operator: bool = False):
    with tempfile.TemporaryDirectory() as workdir:
        download(subset, workdir)
        preprocess(subset, workdir)

        test_file_path = os.path.join(subset, "test.jsonl")
        aug_file_path = os.path.join(subset, "aug_test.jsonl")
        augmenter = None
        match subset:
            case "Python800":
                augmenter = augment_Python800
            case "Java250":
                augmenter = augment_Java250
            case "C++1000" | "C++1400":
                augmenter = augment_Cpp
            case _:
                logging.info(f"Skipping augmentation for {subset}")
                return

        assert augmenter is not None
        augmenter(
            input_file_path=test_file_path,
            output_file_path=aug_file_path,
        )
        if not per_operator:
            return

        for operator_key in get_operator_keys(subset):
            per_op_file = os.path.join(subset, f"aug_test_{operator_key}.jsonl")
            augmenter(
                input_file_path=test_file_path,
                output_file_path=per_op_file,
                operator_key=operator_key,
            )


def main(subset: str, per_operator: bool = False):
    assert subset in ["C++1000", "C++1400", "Java250", "Python800", "all"]

    if subset == "all":
        for s in ["C++1000", "C++1400", "Java250", "Python800"]:
            process(s, per_operator=per_operator)
    else:
        process(subset, per_operator=per_operator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
