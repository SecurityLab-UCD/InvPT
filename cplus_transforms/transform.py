import os
from dataclasses import asdict
from cplus_transforms.transformations.addassignment import add_assignmenter
from cplus_transforms.transformations.local_rename import local_renamer
from cplus_transforms.transformations.p2add import replace_short_adder
from cplus_transforms.transformations.for_while import for_while_reverser
from cplus_transforms.transformations.while_for import while_for_reverser
from cplus_transforms.transformations.if_else_transform import if_else_reverser
from cplus_transforms.transformations.addassignment_cplus import add_assignmenter_cplus
from cplus_transforms.transformations.p2add_cplus import replace_short_adder_cplus
from modeling.dataloader import CodeSearchNetExample, AugType
import ast
import json
import subprocess
from returns.maybe import Maybe, Nothing, Some
from returns.pointfree import bind
import tempfile
import argparse
from typing import Type
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import clang
from collections.abc import Callable

clang.cindex.Config.set_library_file('/usr/lib/llvm-15/lib/libclang.so.1')

TRANSFORMATION_MAP_N: dict[AugType, Callable] = {
    AugType.LOCALVARRENAMING: local_renamer,
    AugType.REVERSEIFELSE: if_else_reverser,
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT: add_assignmenter,
    AugType.PP2ADDASSIGNMENT: replace_short_adder,
    AugType.WHILE2FOR: while_for_reverser,
    AugType.FOR2WHILE: for_while_reverser,
}

TRANSFORMATION_MAP: dict[AugType, Callable] = {
    AugType.LOCALVARRENAMING: local_renamer,
    AugType.REVERSEIFELSE: if_else_reverser,
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT: add_assignmenter_cplus,
    AugType.PP2ADDASSIGNMENT: replace_short_adder_cplus,
    AugType.WHILE2FOR: while_for_reverser,
    AugType.FOR2WHILE: for_while_reverser,
}

def apply_code_transformation(naive: bool, aug_type: AugType, code: str) -> Maybe[str]:
    """
    Given an augmentation type and source code, return the transformed code
    """
    try:
        tmap = TRANSFORMATION_MAP
        if naive:
            tmap = TRANSFORMATION_MAP_N
        ast_transformer = tmap[aug_type]
        index = clang.cindex.Index.create()
        translation_unit = index.parse('example.cpp', unsaved_files=[('example.cpp', code)], options=0)
        return Some(ast_transformer(translation_unit.cursor, code))
    except:
        print("Error Occured")
        return Nothing


def transform(csn_example: CodeSearchNetExample) -> Maybe[CodeSearchNetExample]:
    """
    Apply the ast.NodeTransformer class on the source code and return the transformed code
    """
    csn_example.transformed = apply_code_transformation(
        csn_example.aug_type, csn_example.code
    )
    return csn_example


def load_csn_example(augtype: AugType, json_line: str) -> Maybe[CodeSearchNetExample]:
    data = json.loads(json_line)
    try:
        csn_example = CodeSearchNetExample(
            repo=data["repo"],
            func_name=data["func_name"],
            language=data["language"],
            code=data["code"],
            docstring=data["docstring"],
            transformed="",
            aug_type=augtype,
        )
    except KeyError:
        return Nothing

    return Some(csn_example)


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
                case Some(csn):
                    f.write(json.dumps(asdict(csn)) + "\n")
                case Nothing:
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