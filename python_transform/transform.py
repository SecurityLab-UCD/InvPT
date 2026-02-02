import argparse
import ast
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from functools import partial
from multiprocessing import cpu_count
from typing import Type

from pathos.multiprocessing import ProcessingPool as Pool
from returns.maybe import Maybe, Nothing, Some
from returns.pointfree import bind

from modeling.dataloader import AugType, CodeSearchNetExample
from python_transform.src import (
    LocalVariableRenamer,
    OpAssignment2EqualAssignment,
    ReverseIfElser,
)

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

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".py", delete=True) as temp:
        temp.write(source_code.encode())
        temp.flush()
        try:
            subprocess.run(
                ["2to3", temp.name, "-w"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            return Nothing
        except TypeError:
            # while trying to convert the code from py2 to py3, it will throw an error if the code ifself has syntax error
            # in this case, we skip this transformation
            return Nothing

        # Read the modified content from the file
        temp.seek(0)
        modified_code = temp.read().decode()
        return Some(modified_code)


def parse(source_code: str) -> Maybe[ast.Module]:
    original_ast_module: Maybe[ast.Module]
    try:
        original_ast_module = Some(ast.parse(source_code))
    except SyntaxError:
        try:
            original_ast_module = convert_python2_to_python3(source_code).map(ast.parse)
        except SyntaxError:
            return Nothing
    except RecursionError:
        return Nothing
    return original_ast_module


def apply_code_transformation(aug_type: AugType, code: str) -> Maybe[str]:
    """
    Given an augmentation type and source code, return the transformed code
    """

    def transformer(node: ast.AST) -> Maybe[str]:
        ast_transformer = TRANSFORMATION_MAP[aug_type]()

        # ! fix: ReverseIfElse transformation may introduce RecursionError
        try:
            transformed_node = ast_transformer.visit(node)
            code = ast.unparse(transformed_node)
        except RecursionError:
            return Nothing
        return Some(code)

    return parse(code).bind(transformer)


def transform_csn(csn_example: CodeSearchNetExample) -> Maybe[CodeSearchNetExample]:
    """
    Apply the ast.NodeTransformer class on the source code and return the transformed code
    """

    def csn_add_transformed(transformed: str) -> CodeSearchNetExample:
        return CodeSearchNetExample(
            repo=csn_example.repo,
            func_name=csn_example.func_name,
            language=csn_example.language,
            code=csn_example.code,
            docstring=csn_example.docstring,
            transformed=transformed,
            aug_type=csn_example.aug_type,
        )

    assert csn_example.aug_type is not None, "Augmentation type is not provided"
    return apply_code_transformation(csn_example.aug_type, csn_example.code).map(
        csn_add_transformed
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

    assert os.path.exists(input_file_path) and os.path.isfile(input_file_path), (
        "Invalid input file path"
    )

    # read in the jsonl file
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    with Pool(num_cpus) as pool:
        csn_examples = pool.map(partial(load_csn_example, augtype), lines)
        transformed_data = pool.map(bind(transform_csn), csn_examples)

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
