# Combine augmented.jsonl and data.jsonl

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
        prog='merge',
        description="Merge data.jsonl and augmented.jsonl")
parser.add_argument('--augmented_path', required=True)
parser.add_argument('--original_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

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
        print("read complete! Here's a preview")
        print(df.head(3))
        return df

AUG_PATH = Path(args.augmented_path)
ORIG_PATH = Path(args.original_path)
OUTPUT_PATH = Path(args.output_path)

data_df = jsonl_to_df(ORIG_PATH)
aug_df = jsonl_to_df(AUG_PATH)

data_df['aug_type'] = np.nan
data_df = data_df.assign(
    aug_type = np.nan,
    before_idx = data_df['idx'],
    before = np.nan,
)

aug_df = aug_df.rename(columns={
    'func': 'before',
    'transformed': 'func',
    'idx': 'before_idx',
})
idx = data_df.max()['idx'] + 1
aug_df = aug_df.assign(idx=pd.Series(np.arange(idx, idx + aug_df.shape[0])).values)

def strip_func(row):
    func = row['func']
    row['func'] = func.split('\n', 1)[1].rsplit('\n', 1)[0]
    return row
aug_df = aug_df.apply(strip_func, axis=1)

combined_df = pd.concat([data_df, aug_df])
combined_df.to_json(OUTPUT_PATH, orient='records', lines=True)

