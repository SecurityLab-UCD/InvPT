from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import os
import pandas as pd
import shutil
import subprocess

parser = argparse.ArgumentParser(
    prog="augment.py", description="Augment a Java dataset"
)
parser.add_argument("--input_path", required=True, help="Path to the input jsonl")
parser.add_argument(
    "--output_path", required=True, help="Path to save the augmented jsonl"
)

parser.add_argument(
    "--spat_jar", default="SPAT-linux.jar", help="Path to SPAT-linux.jar"
)
parser.add_argument(
    "--spat_lib",
    default="/usr/lib/jvm/java-18-openjdk-amd64/lib",
    help="Path to Java library",
)
parser.add_argument(
    "--rules", nargs="*", type=int, default=[0, 1, 2, 3, 6, 7], help="SPAT rules to use"
)


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


def decompose(original_df, code_dir):
    """Decompose a dataframe into java files to be processed by SPAT.

    The java files will have names n<idex>.java, where <idex> corresponds to the
    original_df.index column.
    """
    max_idlen = len(str(original_df["index"].max()))
    for _, entry in tqdm(original_df.iterrows(), desc="Decomposing data"):
        idstr = str(entry["index"]).zfill(max_idlen)
        java_path = code_dir / f"n{idstr}.java"
        entry.code = f"class n{idstr}{{\n{entry.code}\n}}"
        with open(java_path, "w") as f:
            f.write(entry.code)


def postprocess(
    original: pd.DataFrame, transformed_path: Path, rule_id: int
) -> pd.DataFrame:
    """Process SPAT output

    Arguments:
    original -- Unaugmented code. (index: int, code: str, label: Any)
    transforme_path -- The path to SPAT output java files
    rule_id -- the ID of the rule for transformation

    Returns: Augmented dataframe (code: str, label: Any, aug_type: str, success:
    bool, (index) aug_from: int)
    """
    augmented = pd.DataFrame(original)
    augmented["aug_type"] = pd.Series(np.full(augmented.shape[0], id_to_name[rule_id]))
    augmented["success"] = pd.Series(np.full(augmented.shape[0], False))
    augmented = augmented.rename({"index": "aug_from"}, axis="columns")
    augmented = augmented.set_index("aug_from")

    for file in tqdm(os.listdir(transformed_path)):
        aug_from = int(file.lstrip("n").rstrip(".java"))
        with open(transformed_path / file) as f:
            transformed = f.read()
        augmented.loc[aug_from, "code"] = transformed
        augmented.loc[aug_from, "success"] = True
    return augmented


def spat(
    original: pd.DataFrame, spat_jar: Path, rule_ids: list[int], lib_path: Path
) -> pd.DataFrame:
    """Run SPAT on `original`, returning a DataFrame containing
    augmented entries.

    If an entry of `original` cannot be augmented by a given rule, its
    corresponding entry in the output would contain the original code and
    success=False

    Arguments:
    original -- Unaugmented code. (index: int, code: str, label: Any)
    spat_jar -- Path to the SPAT jarfile
    rule_ids -- the IDs of the augmentation rule; see README
    lib_path -- the library used by SPAT

    Returns: Dataframe (index: int, code: str, label: Any, aug_type: str,
    success: bool, aug_from: int)
    """
    artifact_path = Path("tmp")
    transformed_path = artifact_path / Path("transformed")
    original_path = artifact_path / Path("original")
    os.mkdir(artifact_path)
    os.mkdir(original_path)

    dfs = []
    decompose(original, artifact_path / "original")
    for rule_id in rule_ids:
        print(f"Augmenting dataset with rule {rule_id}...")
        subprocess.run(
            [
                "java",
                "-jar",
                spat_jar,
                str(rule_id),
                original_path,
                transformed_path,
                lib_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dfs.append(postprocess(original, transformed_path, rule_id))
        shutil.rmtree(transformed_path)
    shutil.rmtree(artifact_path)

    original["aug_type"] = pd.Series(np.full(original.shape[0], "None"))
    original["success"] = pd.Series(np.full(original.shape[0], True))
    original["aug_from"] = original["index"]
    index = original["index"].max() + 1
    for i in range(0, len(dfs)):
        dfs[i]["index"] = range(index, index + len(dfs[i]))
        dfs[i] = dfs[i].reset_index()
        index += len(dfs[i])
    output = pd.concat([original, *dfs], ignore_index=True)

    return output


if __name__ == "__main__":
    args = parser.parse_args()
    original = jsonl_to_df(args.input_path)
    assert set(["label", "index", "code"]).issubset(original.columns)
    shutil.copyfile(args.input_path, args.output_path)
    df_result = spat(original, args.spat_jar, args.rules, args.spat_lib)
    print(df_result)
    with open(args.output_path, "w") as f:
        f.write(df_result.to_json(orient="records", lines=True, force_ascii=False))
