# Generate new tests from old tests and augmented cases

import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
        prog='aug_test.py',
        description="Augment test cases of BCB")
parser.add_argument('--augmented_jsonl', required=True)
parser.add_argument('--test_txt', required=True)
parser.add_argument('--output_txt', required=True)
args = parser.parse_args()

AUG_DATA_PATH = Path(args.augmented_jsonl)
TEST_PATH = Path(args.test_txt)
OUTPUT_PATH = Path(args.output_txt)

def jsonl_to_df(path, chunksize=1000):
    with open(path, 'r') as file:
        # Count total lines in the file
        total_lines = sum(1 for _ in file)

    with open(path, 'r') as file, tqdm(total=total_lines, desc=f'reading {path}') as pbar:
        chunks = []
        for chunk in pd.read_json(file, lines=True, chunksize=chunksize):
            chunks.append(chunk)
            pbar.update(chunksize)
        df = pd.concat(chunks, ignore_index=True)
        return df

df = jsonl_to_df(AUG_DATA_PATH)

with open(OUTPUT_PATH, 'w') as out_file, open(TEST_PATH, 'r') as in_file:
    for line in tqdm(in_file):
        line = line.split()
        idx1 = line[0]
        idx2 = line[1]
        is_similar = line[2]

        idx1_subs = df.loc[df['before_idx'] == int(idx1)]['idx']
        idx2_subs = df.loc[df['before_idx'] == int(idx2)]['idx']
        combinations = [(idx1,idx2) for idx1 in idx1_subs for idx2 in idx2_subs]
        for idx1, idx2 in combinations:
            out_file.write(f"{idx1}\t{idx2}\t{is_similar}\n")

