import os
from dataclasses import asdict
from modeling.dataloader import CodeSearchNetExample, AugType
from cpp_transforms.transformations import (
    local_renamer,
    for_while_reverser,
    while_for_reverser,
    if_else_reverser,
    add_assignmenter,
    replace_short_adder,
)
import json
from returns.result import ResultE, safe, Success, Failure
from returns.pointfree import bind
import argparse
from typing import Type, Any
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import clang
import clang.cindex
from clang.cindex import Index as CursorIndex, Cursor
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

class HasCode(Protocol):
    code: str

class HasFunc(Protocol):
    func: str

TRANSFORMATION_MAP: dict[AugType, Callable[[Cursor, str], str]] = {
    AugType.LOCALVARRENAMING: local_renamer,
    AugType.REVERSEIFELSE: if_else_reverser,
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT: add_assignmenter,
    AugType.PP2ADDASSIGNMENT: replace_short_adder,
    AugType.WHILE2FOR: while_for_reverser,
    AugType.FOR2WHILE: for_while_reverser,
}


@safe
def apply_code_transformation(aug_type: AugType, code: str) -> str:
    """
    Given an augmentation type and source code, return the transformed code
    """
    ast_transformer = TRANSFORMATION_MAP[aug_type]
    index = CursorIndex.create()
    translation_unit = index.parse(
        "example.cpp", unsaved_files=[("example.cpp", code)], options=0
    )
    return ast_transformer(translation_unit.cursor, code)

def augment_accumulatively(j: HasCode | HasFunc) -> HasCode | HasFunc:
    # Check what type of object
    if hasattr(j, 'code'):
        code = j.code
    elif hasattr(j, 'func'):
        code = j.func
    # Run the transformations on the code
    for aug_type in TRANSFORMATION_MAP.keys():
        code = apply_code_transformation(aug_type, code).value_or(code)
    # Create the return object
    ret_obj = replace(j)
    if hasattr(j, 'code'):
        ret_obj.code = code
    elif hasattr(j, 'func'):
        ret_obj.func = code
    return ret_obj

def transform(csn_example: CodeSearchNetExample) -> ResultE[CodeSearchNetExample]:
    """
    Apply the ast.NodeTransformer class on the source code and return the transformed code
    """

    def add_transformed(transformed_code: str) -> CodeSearchNetExample:
        csn_example.transformed = transformed_code
        return csn_example

    assert csn_example.aug_type is not None
    return apply_code_transformation(csn_example.aug_type, csn_example.code).map(
        add_transformed
    )


@safe
def load_csn_example(augtype: AugType, json_line: str) -> CodeSearchNetExample:
    data = json.loads(json_line)
    return CodeSearchNetExample(
        repo=data["repo"],
        func_name=data["func_name"],
        language=data["language"],
        code=data["code"],
        docstring=data["docstring"],
        transformed="",
        aug_type=augtype,
    )


def main(
    augtype: AugType,
    input_file_path: str,
    output_file_path: str,
    num_cpus: int,
):
    print(f"-------- Selected Transforming Method: {augtype} -------- ")

    assert os.path.exists(input_file_path) and os.path.isfile(
        input_file_path
    ), "Invalid input file path"

    # read in the jsonl file
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    with Pool(num_cpus) as pool:
        csn_examples = pool.map(partial(load_csn_example, augtype), lines)
        transformed_data = pool.map(bind(transform), csn_examples)

    with open(output_file_path, "w") as f:
        for transformed_csn in transformed_data:
            match transformed_csn:
                case Success(csn):
                    f.write(json.dumps(asdict(csn)) + "\n")
                case Failure(e):
                    pass

    print("\nFinished Transformed!\n\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # Parsing Arguments
    arg_parser.add_argument(
        "-t",
        "--augtype",
        help="The id of transformation method",
        type=str,
        choices=[k.value for k in TRANSFORMATION_MAP.keys()],
    )
    arg_parser.add_argument(
        "-i",
        "--input_path",
        help="Path to jsonl file for transformation",
        type=str,
    )
    arg_parser.add_argument(
        "-o",
        "--output_path",
        help="The target jsonl file where the transformed code are located",
        type=str,
    )
    arg_parser.add_argument(
        "-n",
        "--num_cpus",
        help="The number of CPU cores to use for parallel processing",
        type=int,
        default=cpu_count(),
    )

    args = arg_parser.parse_args()
    main(AugType(args.augtype), args.input_path, args.output_path, args.num_cpus)
