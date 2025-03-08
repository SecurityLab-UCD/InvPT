from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm
from java_transform.utils import jsonl_to_df, decompose, id_to_name, DIR_PATH
import fire
import numpy as np
import os
import pandas as pd
import re
import subprocess


def rename_augfrom(name: str) -> str:
    """Rename the old augmentation columns

    0 is the current augmentation, 1 is the previous, so on and so forth.
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
    return name


def col_key(col: str) -> tuple[float, str]:
    """Key for sorting the columns"""
    match = re.match(r"(aug_from|aug_success|aug_type)_(\d+)", col)
    if match:
        return int(match.group(2)), match.group(1)  # Sort by number first, then type
    return -1, col  # Put non-matching columns at the start


def postprocess(
    original: pd.DataFrame,
    transformed_path: Path,
    rule_id: int,
    start_index: int,
) -> pd.DataFrame:
    """Process SPAT output

    For all unspecified columns provided to `original`, each augmented entry
    will have the same value on those columns as their original.

    The `index` of the returned DataFrame is the same as its `aug_from_0`
    column.

    Arguments:
    original: Unaugmented code [*: int](code: str).
    transformed_path: The path to SPAT output java files
    rule_id: the ID of the rule for transformation
    start_index: the lowest index of the returned DataFrame

    Returns: Augmented dataframe [*: int](code: str, aug_type_0: str,
    aug_success_0: bool, aug_from_0: int)
    """
    augmented = pd.DataFrame(original)
    augmented = augmented.rename(rename_augfrom, axis="columns")
    augmented["aug_type_0"] = pd.Series(
        np.full(augmented.shape[0], id_to_name[rule_id]),
        index=augmented.index,
    )
    augmented["aug_from_0"] = augmented.index
    augmented["aug_success_0"] = pd.Series(
        np.full(augmented.shape[0], False),
        index=augmented.index,
    )

    for file in tqdm(os.listdir(transformed_path), desc="Processing SPAT output"):
        aug_from = int(file.lstrip("n").rstrip(".java"))
        with open(transformed_path / file) as f:
            transformed = f.read()
        # Strip the wrapper class
        transformed = "\n".join(transformed.splitlines()[1:-1])
        augmented.loc[aug_from, "code"] = transformed
        augmented.loc[aug_from, "aug_success_0"] = True
    augmented.index = range(start_index, start_index + augmented.shape[0])
    return augmented


def spat(
    original: pd.DataFrame,
    spat_jar: Path,
    rule_ids: list[int],
    lib_path: Path,
    include_original: bool,
) -> pd.DataFrame:
    """Run SPAT on `original`, returning a DataFrame containing
    augmented entries.

    If an entry of `original` cannot be augmented by a given rule, its
    corresponding entry in the output would contain the original code and
    aug_success_0=False

    For all unspecified columns provided to `original`, each augmented entry
    will have the same value on those columns as their original.

    The `index` of the returned `DataFrame` starts at the largest `index` of
    `original` + 1.

    Arguments:
    original: Unaugmented code. [*: int](code: str)
    spat_jar: Path to the SPAT jarfile
    rule_ids: the IDs of the augmentation rule; see README
    lib_path: the library used by SPAT

    Returns: Dataframe [*: int](code: str, aug_type_0: str, aug_success_0: bool,
    aug_from_0: int)
    """
    dfs = []
    with TemporaryDirectory() as original_dir:
        decompose(original, Path(original_dir))
        index = original.index.max()
        index = index + 1
        for rule_id in rule_ids:
            print(f"Augmenting dataset with rule {rule_id}...")
            with TemporaryDirectory() as transformed_dir:
                subprocess.run(
                    [
                        "java",
                        "-jar",
                        spat_jar,
                        str(rule_id),
                        original_dir,
                        transformed_dir,
                        lib_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                dfs.append(postprocess(original, Path(transformed_dir), rule_id, index))
                index += dfs[-1].shape[0]

    for i in range(0, len(dfs)):
        name = dfs[i].index.name
        dfs[i].index = range(index, index + len(dfs[i]))
        dfs[i].index.name = name
        index += len(dfs[i])
    if include_original:
        original = original.rename(rename_augfrom, axis="columns")
        original["aug_type_0"] = pd.Series(
            np.full(original.shape[0], "None"),
            index=original.index,
        )
        original["aug_success_0"] = pd.Series(
            np.full(original.shape[0], True),
            index=original.index,
        )
        original["aug_from_0"] = original.index
        dfs.append(original)
    output = pd.concat(dfs)

    return output


def main(
    input_path: str,
    output_path: str,
    spat_jar: str = str(DIR_PATH / "SPAT-linux.jar"),
    spat_lib: str = "/usr/lib/jvm/java-18-openjdk-amd64/lib",
    rules: list[int] = [0, 1, 2, 3, 6, 7],
    include_original: bool = False,
    accumulate: bool = True,
    debug: bool = False,
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
    input_path: Path to the input jsonl (index: int, code: str)
    output_path: Path to save the augmented jsonl (code: str, aug_type_0: str,
        aug_success_0: bool, aug_from_0: int)
    spat_jar: Path to the SPAT jar file used for augmentation
    spat_lib: Path to the Java library used by SPAT (see SPAT documentation)
    rules: List of SPAT rule IDs.
    include_original: If True, the augmented dataset contains the original
    accumulate: If True, the rules are applied accumulatively instead of
        mapping (i.e. [t1(t2(x)]) instead of [t1(x), t2(x)])
    debug: Only augment the first 100 entries
    """
    df = jsonl_to_df(input_path)
    if debug:
        df = df.iloc[:100]
    assert set(["index", "code"]).issubset(df.columns)
    df.set_index("index", inplace=True)
    if accumulate:
        for rule in rules:
            df = spat(df, Path(spat_jar), [rule], Path(spat_lib), include_original)
    else:
        df = spat(df, Path(spat_jar), rules, Path(spat_lib), include_original)
    df.reset_index(inplace=True)
    df = df[sorted(df.columns, key=col_key)]
    with open(output_path, "w") as f:
        f.write(df.to_json(orient="records", lines=True, force_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
