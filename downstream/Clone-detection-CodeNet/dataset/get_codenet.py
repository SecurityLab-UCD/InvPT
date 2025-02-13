from download import main as download
from preprocess import main as preprocess
import fire
import tempfile
import logging


def process(subset: str):
    with tempfile.TemporaryDirectory() as workdir:
        download(subset, workdir)
        preprocess(subset, workdir)


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
