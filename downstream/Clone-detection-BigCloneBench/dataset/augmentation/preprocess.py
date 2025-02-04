from pathlib import Path
import os
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Preprocessing data.jsonl"
                                 "into a bunch of .java files and a"
                                 "metadata.jsonl")

parser.add_argument("--data_jsonl", required=True)
parser.add_argument("--extracted_java_path", required=True, help="Written to")
parser.add_argument("--metadata_jsonl_path", required=True, help="Written to")
parser.add_argument("--java_colname", required=True, help="The name of the column containing the java code")
parser.add_argument("--id_colname", help="The name of the column containing the ID. If not set, uses the row number")
parser.add_argument("--drop_duplicates",
                    action=argparse.BooleanOptionalAction,
                    help="Drop entries that are duplicates in java_colname")

args = parser.parse_args()
DATA_JSONL = Path(args.data_jsonl)
OUTPUT_DIR = Path(args.extracted_java_path)
METADATA_PATH = Path(args.metadata_jsonl_path)
CODE_COL = args.java_colname
ID_COL = args.id_colname
DROP_DUPLICATES = args.drop_duplicates

print(f'reading {DATA_JSONL}...')
with open(DATA_JSONL, 'r') as f:
    df = pd.read_json(f, lines=True)
if DROP_DUPLICATES:
    print('dropping duplicates...')
    df = df.drop_duplicates(subset=CODE_COL).reset_index(drop=True)

try:
    os.mkdir(OUTPUT_DIR)
except Exception as e:
    print(f"WARNNING: Assume directory {OUTPUT_DIR} is valid")

num_uniques = len(df)
print(f'extracting {num_uniques} unique code segments...')
for id, entry in tqdm(df.iterrows()):
    if not ID_COL is None:
        id = entry[ID_COL]
    idstr = str(id).zfill(len(str(num_uniques)))
    java_path = OUTPUT_DIR / f'n{idstr}.java'
    entry[CODE_COL] = f"class n{idstr}{{\n{entry[CODE_COL]}\n}}"
    with open(java_path, "w") as f:
        f.write(entry[CODE_COL])

print('writing metadata...')
df.to_json(METADATA_PATH, orient='records', lines=True)

