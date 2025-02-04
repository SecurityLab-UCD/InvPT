import sys

sys.path.append("../../../")
sys.path.append("../../../python_transform")
import os
import shutil
import random
import json
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Tuple
from python_transform.transform import transform
from returns.maybe import Maybe, Nothing, Some
from tqdm import tqdm

source_root = "Project_CodeNet_Python800"
dataset_jsonl_path = "python800_dataset.jsonl"
JSON_ENCODING = "latin-1"
SPLIT_SEED = 0
TRAIN_PERCENTAGE = 0.5
TEST_PERCENTAGE = 0.25
VALID_PERCENTAGE = 0.25


def get_python_files(source_root: str):
    python_files = []
    for root, _, files in os.walk(source_root):
        for file in files:
            python_files.append(os.path.join(root, file))
    return python_files


def split_and_move(source_root: str, output_dir: str):
    """
    Retrieve all Python files from CodeNet_Python800 and split them into train/test/valid for training
    NOTE: They are Python code instead of three jsonl.
    """

    def move_file(src_dst_pair: Tuple[str, str]):
        """Move the file from src to dest"""
        src, dest = src_dst_pair
        shutil.move(src, dest)

    # Collect all python files
    python_files = get_python_files(source_root)
    total_files = len(python_files)
    print(f"Get all python files with total number == {total_files}")
    print("Splitting...")

    # shuffle with seed == 0
    random.seed(SPLIT_SEED)
    random.shuffle(python_files)

    # split
    train_split = int(TRAIN_PERCENTAGE * total_files)
    test_split = train_split + int(TEST_PERCENTAGE * total_files)

    train_files = python_files[:train_split]
    test_files = python_files[train_split:test_split]
    valid_files = python_files[test_split:]

    # create sub directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    valid_dir = os.path.join(output_dir, "valid")

    for subdirectory in [train_dir, test_dir, valid_dir]:
        os.makedirs(subdirectory, exist_ok=True)

    print(
        f"Saving to: \nTrain: {train_dir} ({len(train_files)} files)\nTest: {test_dir} ({len(test_files)} files)\nValid: {valid_dir} ({len(valid_files)} files)"
    )

    move_tasks = []
    for file in train_files:
        move_tasks.append((file, os.path.join(train_dir, os.path.basename(file))))
    for file in test_files:
        move_tasks.append((file, os.path.join(test_dir, os.path.basename(file))))
    for file in valid_files:
        move_tasks.append((file, os.path.join(valid_dir, os.path.basename(file))))

    num_workers = min(8, cpu_count())
    with Pool(num_workers) as pool:
        pool.map(move_file, move_tasks)

    print("Successfully split all the data!")


def split_dataset(dataset_jsonl: str) -> list[str, str, str]:
    """
    Split the dataset.jsonl into train.jsonl, test.jsonl, and valid.jsonl
    """
    train_path = "train.jsonl"
    test_path = "test.jsonl"
    valid_path = "valid.jsonl"

    with open(dataset_jsonl, "r", encoding=JSON_ENCODING) as f:
        data = [json.loads(line) for line in f]
    total_files = len(data)
    random.seed(SPLIT_SEED)
    random.shuffle(data)

    # Compute split sizes
    train_split = int(TRAIN_PERCENTAGE * total_files)
    test_split = train_split + int(TEST_PERCENTAGE * total_files)

    train_data = data[:train_split]
    test_data = data[train_split:test_split]
    valid_data = data[test_split:]

    # Save the splits
    for split_path, split_data in zip(
        [train_path, test_path, valid_path], [train_data, test_data, valid_data]
    ):
        with open(split_path, "w", encoding=JSON_ENCODING) as f_out:
            for entry in split_data:
                f_out.write(json.dumps(entry) + "\n")

    print(
        f"num of files: train= {len(train_data)}, test= {len(test_data)}, valid= {len(valid_data)}"
    )
    return [train_path, test_path, valid_path]


def python800_to_jsonl(dataset_root: str):
    """
    create one jsonl file from Python800 dataset
    returns the jsonl file path
    """
    python_files = get_python_files(dataset_root)
    total_files = len(python_files)

    with open(dataset_jsonl_path, "w") as f:
        for file in python_files:
            # Convert the current python file into JSON object
            js = {}
            js["code"] = open(file, encoding=JSON_ENCODING).read()
            # Write as a Json Line
            f.write(json.dumps(js) + "\n")


def add_augmented_data(test_jsonl: str):
    initial_size: int
    final_size: int

    with open(test_jsonl, "r", encoding=JSON_ENCODING) as f:
        # store the while jsonl file into a list; each item is a sample in testset
        list_of_json = [json.loads(json_line) for json_line in f]

    final_size = initial_size = len(list_of_json)

    for i in tqdm(range(initial_size), desc="adding transformed files..."):

        code = list_of_json[i]["code"]
        transformed_codes = transform(code)
        if transformed_codes == Nothing:
            continue

        with open(test_jsonl, "a") as f:
            for transformed_code in transformed_codes:
                js = {}
                js["code"] = transformed_code
                f.write(json.dumps(transformed_code))
        final_size += 1

    print(f"intial test.jsonl size: {initial_size}; final size: {final_size}")


def preprocess_dataset(dataset_path: str):
    # convert Python800 to JSONL
    python800_to_jsonl(dataset_path)
    # # Split JSONL to train/valid/test for clarity
    _, test_path, _ = split_dataset(dataset_jsonl_path)
    # # In test, add new transformed code
    add_augmented_data(test_path)


# split_and_move(source_root, ".")
if __name__ == "__main__":
    preprocess_dataset(source_root)
