import sys

sys.path.append("..")
import os
from dataclasses import asdict
from src import (
    LocalVariableRenamer,
    ReverseIfElser,
    OpAssignment2EqualAssignment,
)
from modeling.dataloader import CodeSearchNetExample, AugType
import ast
import json
import subprocess
from returns.maybe import Maybe, Nothing, Some
from returns.pointfree import bind
import tempfile
import argparse
from typing import Type, Tuple
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial, singledispatch
from pathlib import Path
import copy


TRANSFORMATION_MAP: dict[AugType, Type[ast.NodeTransformer]] = {
    AugType.LOCALVARRENAMING: LocalVariableRenamer,
    AugType.REVERSEIFELSE: ReverseIfElser,
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT: OpAssignment2EqualAssignment,
}

def convert_source_to_ast_module(source_code: str) -> Maybe[ast.Module]:
    original_ast_module: Maybe[ast.Module]
    try:        
        original_ast_module = Some(ast.parse(source_code))
    except SyntaxError: 
        original_ast_module = convert_python2_to_python3(source_code).map(ast.parse)
    except RecursionError as e:
        return Nothing
    return original_ast_module

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
        return Nothing
    except TypeError as e:
        # while trying to convert the code from py2 to py3, it will throw an error if the code ifself has syntax error
        # in this case, we skip this transformation
        return Nothing
    except RecursionError as e:
        return Nothing

    # Read the modified content from the file
    modified_code = temp.read().decode()
    temp.close()
    return Some(modified_code)


@singledispatch
def transform(origional_example):
    raise NotImplementedError("Unsupported type!")

@transform.register
def _(py_file_path: str) -> Maybe[Tuple[str, str, str]]:
    """
    Apply three ast.NodeTransformers on the source code, and return three transformed codes.
    Returns `Nothing` if all transformations fail, otherwise returns a tuple of three.
    """
    ast_localVariableRenamer = LocalVariableRenamer()
    ast_reverseIfElse = ReverseIfElser()
    ast_opAss2EqualAss = OpAssignment2EqualAssignment()

    with open(py_file_path) as f:
        lines = f.readlines()
        source_code = ''.join(lines)

    def apply_transformation(transformer: ast.NodeTransformer) -> Maybe[str]:        
        original_ast_module: Maybe[ast.Module] = convert_source_to_ast_module(source_code)
        if original_ast_module == Nothing: return Nothing
        try:
            return original_ast_module.map(transformer.visit).map(ast.unparse)
        except RecursionError:
            return Nothing

    # Apply all transformations
    transformed_results: List[Maybe[str]] = [apply_transformation(TRANSFORMATION_MAP[k]()) for k in TRANSFORMATION_MAP.keys()]
    
    # If all transformations fail, return Nothing
    if all(result == Nothing for result in transformed_results): return Nothing

    # If some of results are None (when an error is raised while parsing AST tree), mark it as None
    return Some(tuple(result.value_or(None) for result in transformed_results))

@transform.register
def _(origional_example: CodeSearchNetExample) -> Maybe[CodeSearchNetExample]:
    """
    Apply the ast.NodeTransformer class on the source code and return the transformed code

    """
    # parse the source code into AST 

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
    except RecursionError as e:
        return Nothing

    def add_transformed_code(transformed_code: str):
        origional_example.transformed = transformed_code
        return origional_example

    # ToDo: check if `source` is really modified. If not, return Nothing
    try:
        return (
        original_ast_module.map(ast_transformer.visit)
        .map(ast.unparse)
        .map(add_transformed_code)
    )
    except RecursionError as e:
        return Nothing


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
        help="Path to py/jsonl file for transformation",
        type=str,
    )
    arg_parser.add_argument(
        "-o",
        "--output_path",
        help="The target py/jsonl file where the transformed code are located",
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