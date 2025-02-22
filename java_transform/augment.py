from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import pandas as pd
import shutil
import subprocess

parser = argparse.ArgumentParser(prog='augment.py', description='Augment a Java dataset')
parser.add_argument('--input_path', required=True, help='Path to the input jsonl')
parser.add_argument('--output_path', required=True, help='Path to save the augmented jsonl')

parser.add_argument('--spat_jar', default='SPAT-linux.jar', help='Path to SPAT-linux.jar')
parser.add_argument('--spat_lib', default='/usr/lib/jvm/java-18-openjdk-amd64/lib', help='Path to SPAT-linux.jar')
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

id_to_name = [
    'LocalVarRenaming',
    'For2While',
    'While2For',
    'ReverseIfElse',
    'SingleIF2ConditionalExp',
    'ConditionalExp2SingleIF',
    'PP2AddAssignment',
    'AddAssignemnt2EqualAssignment',
    'InfixExpressionDividing',
    'IfDividing',
    'StatementsOrderRearrangement',
    'LoopIfContinue2Else',
    'VarDeclarationMerging',
    'VarDeclarationDividing',
    'SwitchEqualSides',
    'SwitchStringEqual',
    'PrePostFixExpressionDividing',
    'Case2IfElse',
]

def decompose(original_df, code_dir):
    """Decompose a dataframe into java files to be processed by SPAT.

    The java files will have names n<idex>.java, where <idex> corresponds to the
    original_df.index column.
    """
    max_idlen = len(str(original_df.index.argmax()))
    for _, entry in tqdm(original_df.iterrows(), desc="Decomposing data"):
        idstr = str(entry["index"]).zfill(max_idlen)
        java_path = code_dir / f'n{idstr}.java'
        entry.code = f"class n{idstr}{{\n{entry.code}\n}}"
        with open(java_path, "w") as f:
            f.write(entry.code)



def spat(spat_jar, df, rule_ids, lib_path, output_path):
    """Run SPAT on all entries in the dataframe and append the results to
    output_path

    The original (unaugmented) entries are not written
    """
    artifact_path = Path('tmp')
    transformed_path = artifact_path / Path('transformed')
    os.mkdir(artifact_path)
    os.mkdir(artifact_path / 'original')

    new_id = int(df.index.argmax()) + 1
    decompose(df, artifact_path / 'original')
    for rule_id in rule_ids:
        print(f'Augmenting dataset with rule {rule_id}...')
        subprocess.run(
            ["java", "-jar", spat_jar, str(rule_id), artifact_path / 'original',
                 transformed_path, lib_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        print('Saving results')
        for file in tqdm(os.listdir(transformed_path)):
            code_id = int(file.lstrip('n').rstrip('.java'))
            with open(transformed_path / file) as f:
                transformed = f.read()
            entry = df.loc[df["index"] == code_id].iloc[0].to_dict()
            entry = {
                'index': new_id,
                'label': entry['label'],
                'code': transformed,
                'aug_type': id_to_name[rule_id],
                'aug_from': entry["index"]
            }
            new_id += 1
            with open(output_path, 'a') as f:
                f.write(f'{json.dumps(entry)}\n')
        shutil.rmtree(transformed_path)
    shutil.rmtree(artifact_path)

if __name__ == "__main__":
    args = parser.parse_args()
    original = jsonl_to_df(args.input_path)
    assert set(['label', 'index', 'code']).issubset(original.columns)
    shutil.copyfile(args.input_path, args.output_path)
    spat(args.spat_jar, original, args.rules, args.spat_lib, args.output_path)
