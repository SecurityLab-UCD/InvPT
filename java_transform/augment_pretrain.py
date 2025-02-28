import fire
from pathlib import Path
from augment import spat, jsonl_to_df, DIR_PATH


def col_key(col: str) -> float:
    """Key for sorting the columns"""
    match col:
        case "code":
            return 0
        case "transformed":
            return 1
        case "aug_type":
            return 2

    return float("inf")


def main(
    input_path: str,
    output_path: str,
    spat_jar: str = str(DIR_PATH / "SPAT-linux.jar"),
    spat_lib: str = "/usr/lib/jvm/java-18-openjdk-amd64/lib",
    rules: list[int] = [0, 1, 2, 3, 6, 7],
    include_original: bool = True,
):
    """Augment pretrain CodeSearchNet dataset

    Args:
    input_path: Path to the input jsonl (code: str)
    output_path: Path to save the augmented jsonl (code: str, transformed: str,
        aug_type: str)
    spat_jar: Path to the SPAT jar file used for augmentation
    spat_lib: Path to the Java library used by SPAT (see SPAT documentation)
    rules: List of SPAT rule IDs.
    include_original: If True, the augmented dataset contains the original
    accumulate: If True, the rules are applied accumulatively instead of
        mapping (i.e. [t1(t2(x)]) instead of [t1(x), t2(x)])
    """
    original = jsonl_to_df(input_path)

    augmented = spat(original, Path(spat_jar), rules, Path(spat_lib), include_original)
    augmented = augmented.loc[augmented["aug_success_0"]]
    augmented.rename(
        {"aug_type_0": "aug_type", "code": "transformed", "aug_from_0": "code"},
        axis="columns",
        inplace=True,
    )
    augmented["code"] = augmented["code"].apply(lambda id: original.loc[id, "code"])
    augmented.drop(["aug_success_0"], axis="columns", inplace=True)

    augmented = augmented[sorted(augmented.columns, key=col_key)]
    with open(output_path, "w") as f:
        f.write(augmented.to_json(orient="records", lines=True, force_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
