from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm
from utils import jsonl_to_df, decompose, id_to_name, DIR_PATH
import numpy as np
import os
import json
import fire
import pandas as pd
from multiprocessing import cpu_count
from java_transform.transform import TRANSFORMATION_MAP, main as transform_main


def main(
    input_path: str,
):
    nproc = cpu_count()
    for augtype in TRANSFORMATION_MAP.keys():
        output_path = f"java_{augtype}.jsonl"
        transform_main(augtype, input_path, output_path, nproc)


if __name__ == "__main__":
    fire.Fire(main)
