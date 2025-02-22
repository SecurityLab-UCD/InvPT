import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='augment.py', description='Augment a Java dataset')
parser.add_argument('--input_path', required=True, help='Path to the input jsonl')
parser.add_argument('--output_path', required=True, help='Path to save the augmented jsonl')

parser.add_argument('--spat_jar', default='SPAT-linux.jar', help='Path to SPAT-linux.jar')
parser.add_argument('--rules', nargs='*', default=[0,1,2,3,6,7], help='SPAT rules to use')

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

def decompose(original_df, save_dir):
    """Decompose a dataframe into java files to be processed by SPAT"""
    max_idlen = len(str(original_df.id.argmax()))
    for _, entry in tqdm(original_df.iterrows(), desc="Decomposing data"):
        idstr = str(entry.id).zfill(max_idlen)
        java_path = save_dir / f'n{idstr}.java'
        entry.code = f"class n{idstr}{{\n{entry.code}\n}}"
        with open(java_path, "w") as f:
            f.write(entry.code)


if __name__ == "__main__":
    args = parser.parse_args()
    original = jsonl_to_df(args.input_path)
    assert set(['label', 'index', 'code']).issubset(original.columns)
