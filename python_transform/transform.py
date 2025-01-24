import sys
sys.path.append('..')
import os
from dataclasses import asdict
from src import (
    LocalVariableRenamer,
    ReverseIfElser,
    OpAssignment2EqualAssignment,
)
from modeling.types import CodeSearchNetExample, AugType
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


TRANSFORMATION_MAP: dict[AugType, Type[ast.NodeTransformer]] = {
    AugType.LOCALVARRENAMING: LocalVariableRenamer,
    AugType.REVERSEIFELSE: ReverseIfElser,
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT: OpAssignment2EqualAssignment,
}


def convert_python2_to_python3(source_code: str) -> Maybe[str]:
    """convert python 2.x syntax to python 3.x syntax

    Args:
        source_code (str): the source code in python 2.x syntax

    Returns:
        Maybe[str]: the source code in python 3.x syntax
    """
    # lib2to3 is no longer in python 3.11; however, we can still use the 2to3 command line!

    temp = tempfile.TemporaryFile()
    temp.write(source_code.encode())
    try:
        subprocess.run(
            ["2to3", temp.name, "-w"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        # TODO: see if there's any other way to solve this issues...
        return Nothing
    except TypeError as e:
        return Nothing

    # Read the modified content from the file
    modified_code = temp.read().decode()
    temp.close()
    return Some(modified_code)


def transform(origional_example: CodeSearchNetExample) -> Maybe[CodeSearchNetExample]:
    """
    Apply the ast.NodeTransformer class on the source code and return the transformed code

    """
    # parse the source code into AST
    # print('source code: ', source)

    source = origional_example.code
    transform_type = origional_example.aug_type

    ast_transformer = TRANSFORMATION_MAP[transform_type]()

    original_ast_module: Maybe[ast.Module]
    try:
        # If Python 3.x syntax, then no error will be raised
        original_ast_module = Some(ast.parse(source))
    except SyntaxError:
        # Ohterwise, convert to pyhon 3.x syntax from python 2.x
        original_ast_module = convert_python2_to_python3(source).map(ast.parse)

    def add_transformed_code(transformed_code: str):
        origional_example.transformed = transformed_code
        return origional_example

    # ToDo: check if `source` is really modified. If not, return Nothing
    return (
        original_ast_module.map(ast_transformer.visit)
        .map(ast.unparse)
        .map(add_transformed_code)
    )


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
