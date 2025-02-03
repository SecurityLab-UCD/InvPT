import sys

sys.path.append("../../../")
sys.path.append("../../../python_transform")
import os
import shutil
import random
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Tuple
from python_transform.transform import transform
from returns.maybe import Maybe, Nothing, Some

source_root = "Project_CodeNet_Python800"
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

"""
create renaming, reverseIfElse, op2equal directory
"""
def transform_test_files(test_dir: str):
    python_files = get_python_files(test_dir)
    total_files = len(python_files)

    single_file_for_testing = python_files[0]
    transformed_code = transform(single_file_for_testing)

    if transformed_code == Nothing: return    
    transformed_code = transformed_code.unwrap()

    
    print("transformed 1")    
    print(transformed_code[0])
    print("transformed 2")    
    print(transformed_code[1])
    print("transformed 3")    
    print(transformed_code[2])

    # TODO: if everything so far are accurate
    # then the next step is: for each file, create three transformed code and put inside the test directory
    

# split_and_move(source_root, ".")
transform_test_files('test/')
