from tqdm import tqdm
from pathlib import Path
import pandas as pd

DIR_PATH = Path(__file__).resolve().parent

id_to_name = [
    "LocalVarRenaming",
    "For2While",
    "While2For",
    "ReverseIfElse",
    "SingleIF2ConditionalExp",
    "ConditionalExp2SingleIF",
    "PP2AddAssignment",
    "AddAssignemnt2EqualAssignment",
    "InfixExpressionDividing",
    "IfDividing",
    "StatementsOrderRearrangement",
    "LoopIfContinue2Else",
    "VarDeclarationMerging",
    "VarDeclarationDividing",
    "SwitchEqualSides",
    "SwitchStringEqual",
    "PrePostFixExpressionDividing",
    "Case2IfElse",
]


def jsonl_to_df(path, chunksize=1000):
    with open(path, "r") as file:
        # Count total lines in the file
        total_lines = sum(1 for _ in file)

    with open(path, "r") as file, tqdm(
        total=total_lines, desc=f"reading {path}"
    ) as pbar:
        chunks = []
        for chunk in pd.read_json(file, lines=True, chunksize=chunksize):
            chunks.append(chunk)
            pbar.update(chunksize)
        df = pd.concat(chunks, ignore_index=True)
        print("read complete! Here's a preview")
        print(df.head(3))
        return df


def decompose(original_df: pd.DataFrame, code_dir: Path):
    """Decompose a dataframe into java files to be processed by SPAT.

    The java files will have names n<idex>.java, where <idex> corresponds to the
    original_df.index column.
    """
    max_idlen = len(str(original_df.index.max()))
    for idx, entry in tqdm(
        original_df.iterrows(), desc="Decomposing data", total=len(original_df)
    ):
        idstr = str(idx).zfill(max_idlen)
        java_path = code_dir / f"n{idstr}.java"
        entry.code = f"class n{idstr}{{\n{entry.code}\n}}"
        with open(java_path, "w") as f:
            f.write(entry.code)
