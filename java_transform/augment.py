from pathlib import Path
from tqdm import tqdm
import re
import fire
import numpy as np
import os
import pandas as pd
import shutil
import subprocess


DIR_PATH = Path(__file__).resolve().parent
print(DIR_PATH)


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


def rename_augfrom(name: str) -> str:
    """Rename the augmentation columns to add a new augmentation.

    0 is the augmentation, 1 is the previous, so on and so forth.
    """
    match = re.match(r"aug_from_(\d+)", name)
    if match:
        return "aug_from_" + str(int(match.group(1)) + 1)
    match = re.match(r"aug_type_(\d+)", name)
    if match:
        return "aug_type_" + str(int(match.group(1)) + 1)
    match = re.match(r"aug_success_(\d+)", name)
    if match:
        return "aug_success_" + str(int(match.group(1)) + 1)
    if name == "index":
        return "aug_from_0"
    return name


def col_key(col: str) -> tuple[float, str]:
    """Key for sorting the columns"""
    match = re.match(r"(aug_from|aug_success|aug_type)_(\d+)", col)
    if match:
        return int(match.group(2)), match.group(1)  # Sort by number first, then type
    return -1, col  # Put non-matching columns at the start


def postprocess(
    original: pd.DataFrame, transformed_path: Path, rule_id: int
) -> pd.DataFrame:
    """Process SPAT output

    For all unspecified columns provided to `original`, each augmented entry
    will have the same value on those columns as their original.

    Arguments:
    original -- Unaugmented code (index: int, code: str).
    transforme_path -- The path to SPAT output java files
    rule_id -- the ID of the rule for transformation

    Returns: Augmented dataframe (code: str, aug_type_0: str, aug_success_0:
    bool, (index) aug_from_0: int)
    """
    augmented = pd.DataFrame(original)
    augmented = augmented.rename(rename_augfrom, axis="columns")
    augmented["aug_type_0"] = pd.Series(np.full(augmented.shape[0], id_to_name[rule_id]))
    augmented["aug_success_0"] = pd.Series(np.full(augmented.shape[0], False))
    augmented = augmented.set_index("aug_from_0")

    for file in tqdm(os.listdir(transformed_path)):
        aug_from = int(file.lstrip("n").rstrip(".java"))
        with open(transformed_path / file) as f:
            transformed = f.read()
        augmented.loc[aug_from, "code"] = transformed
        augmented.loc[aug_from, "aug_success_0"] = True
    return augmented


def spat(
    original: pd.DataFrame, spat_jar: Path, rule_ids: list[int], lib_path: Path,
    include_original: bool,
) -> pd.DataFrame:
    """Run SPAT on `original`, returning a DataFrame containing
    augmented entries.

    If an entry of `original` cannot be augmented by a given rule, its
    corresponding entry in the output would contain the original code and
    aug_success_0=False

    For all unspecified columns provided to `original`, each augmented entry
    will have the same value on those columns as their original.

    Arguments:
    original -- Unaugmented code. (index: int, code: str)
    spat_jar -- Path to the SPAT jarfile
    rule_ids -- the IDs of the augmentation rule; see README
    lib_path -- the library used by SPAT

    Returns: Dataframe (index: int, code: str, aug_type_0: str,
    aug_success_0: bool, aug_from_0: int)
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

    index = original["index"].max() + 1
    for i in range(0, len(dfs)):
        dfs[i]["index"] = range(index, index + len(dfs[i]))
        dfs[i] = dfs[i].reset_index()
        index += len(dfs[i])
    if include_original:
        original = original.rename(rename_augfrom, axis="columns")
        original["aug_type_0"] = pd.Series(np.full(original.shape[0], "None"))
        original["aug_success_0"] = pd.Series(np.full(original.shape[0], True))
        original["index"] = original["aug_from_0"]
        dfs.append(original)
    output = pd.concat(dfs, ignore_index=True)

    return output


def main(
    input_path: str,
    output_path: str,
    spat_jar: str = str(DIR_PATH / "SPAT-linux.jar"),
    spat_lib: str = "/usr/lib/jvm/java-18-openjdk-amd64/lib",
    rules: list[int] = [0, 1, 2, 3, 6, 7],
    include_original: bool = True,
    accumulate: bool = False,
):
    """Augment a Java dataset with SPAT

    Choose from the following rules:
    1. LocalVarRenaming
    2. For2While
    3. While2For
    4. ReverseIfElse
    5. SingleIF2ConditionalExp
    6. ConditionalExp2SingleIF
    7. PP2AddAssignment
    8. AddAssignemnt2EqualAssignment
    9. InfixExpressionDividing
    10. IfDividing
    11. StatementsOrderRearrangement
    12. LoopIfContinue2Else
    13. VarDeclarationMerging
    14. VarDeclarationDividing
    15. SwitchEqualSides
    16. SwitchStringEqual
    17. PrePostFixExpressionDividing
    18. Case2IfElse

    Args:
    input_path: Path to the input jsonl
    output_path: Path to save the augmented jsonl
    spat_jar: Path to the SPAT jar file used for augmentation
    spat_lib: Path to the Java library used by SPAT (see SPAT documentation)
    rules: List of SPAT rule IDs.
    include_original: If True, the augmented dataset contains the original
    accumulate: If True, the rules are applied accumulatively instead of
        mapping (i.e. [t1(t2(x)]) instead of [t1(x), t2(x)])
    """
    df = jsonl_to_df(input_path)
    assert set(["index", "code"]).issubset(df.columns)
    shutil.copyfile(input_path, output_path)
    if accumulate:
        for rule in rules:
            df = spat(df, Path(spat_jar), [rule], Path(spat_lib), include_original)
    else:
        df = spat(df, Path(spat_jar), rules, Path(spat_lib), include_original)
    df = df[sorted(df.columns, key=col_key)]
    print(df)
    with open(output_path, "w") as f:
        f.write(df.to_json(orient="records", lines=True, force_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
