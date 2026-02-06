import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

DIR_PATH = Path(__file__).resolve().parent

PIA_HOME = os.environ.get("PIA_HOME")
if PIA_HOME is None:
    raise EnvironmentError("PIA_HOME environment variable is not set")
SPAT_JAR = os.path.join(PIA_HOME, "java_transform", "SPAT-linux.jar")

_JDK_LIB = os.environ.get("JDK_LIB")
if _JDK_LIB is None:
    raise EnvironmentError("JDK_LIB environment variable is not set")
JDK_LIB: str = _JDK_LIB

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
        return pd.read_json(path, lines=True)


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


Program = tuple[int, str]  # index, code


def write_programs(dst_path: str, programs: list[Program]):
    for i, p in programs:
        class_name = f"p{i}"
        dst_file_path = os.path.join(dst_path, f"{class_name}.java")
        p_w_class = f"class {class_name}{{\n{p}\n}}"
        with open(dst_file_path, "w") as f:
            f.write(p_w_class)


def read_programs(src_path: str) -> list[Program]:
    programs = []
    for file in os.listdir(src_path):
        with open(os.path.join(src_path, file)) as f:
            pid = int(file.lstrip("p").rstrip(".java"))
            code = f.read()
            code = "\n".join(code.splitlines()[1:-1])
            programs.append((pid, code))
    return programs


def spat_caller(
    rule_id: int,
    spat_path: str = SPAT_JAR,
    lib_path: str = JDK_LIB,
) -> Callable[[list[Program]], list[Program]]:
    def transform_dir(input_path: str, output_path: str):
        """Transforms all files in a directory using SPAT.

        Args:
        input_path -- the path to the directory containing the files to transform
        output_path -- the path to save the transformed files
        """
        subprocess.run(
            [
                "java",
                "-jar",
                spat_path,
                str(rule_id),
                input_path,
                output_path,
                lib_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def transform_programs(programs: list[Program]) -> list[Program]:
        with tempfile.TemporaryDirectory() as temp_src:
            with tempfile.TemporaryDirectory() as temp_dst:
                write_programs(temp_src, programs)
                transform_dir(temp_src, temp_dst)
                transformed = read_programs(temp_dst)
        return transformed

    return transform_programs
