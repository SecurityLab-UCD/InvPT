from download import main as download
from preprocess import main as preprocess
import fire
import tempfile
import logging
import os

from augment_Python800 import main as augment_Python800
from augment_Java250 import main as augment_Java250


def process(subset: str):
    with tempfile.TemporaryDirectory() as workdir:
        download(subset, workdir)
        preprocess(subset, workdir)

        test_file_path = os.path.join(subset, "test.jsonl")
        aug_file_path = os.path.join(subset, "aug_test.jsonl")
        match subset:
            case "Python800":
                augment_Python800(
                    input_file_path=test_file_path,
                    ouput_file_path=aug_file_path,
                )
            case "Java250":
                augment_Java250(
                    input_file_path=test_file_path,
                    output_file_path=aug_file_path,
                )
            case _:
                logging.info(f"Skipping augmentation for {subset}")


def main(subset: str):
    assert subset in ["C++1000", "C++1400", "Java250", "Python800", "all"]

    if subset == "all":
        for s in ["C++1000", "C++1400", "Java250", "Python800"]:
            process(s)
    else:
        process(subset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
