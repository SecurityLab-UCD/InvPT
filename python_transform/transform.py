from enum import Enum
from typing import Dict
from dataclasses import dataclass
from src import (
    LocalVariableRenamer,
    FunctionDefinitionReorder,
    ReverseIfElser,
    StatementOrderRearrangement,
    WhileToForTransformer,
    ForToWhileTransformer,
    OpAssignment2EqualAssignment,
    AugType,
)
from modeling.dataloader import CodeSearchNetExample
import ast
import sys
import os
import json
import jsonlines
import autopep8
import subprocess
import os

code_transform_map = {
    0: LocalVariableRenamer,
    1: FunctionDefinitionReorder,
    2: ReverseIfElser,
    3: StatementOrderRearrangement,
    4: OpAssignment2EqualAssignment,
    5: WhileToForTransformer,
    6: ForToWhileTransformer,
}


def convert_python2_to_python3(source_code: str, filename: str = "random.py") -> str:
    # lib2to3 is no longer in python 3.11; however, we can still use the 2to3 command line!
    with open(filename, "w") as file:
        file.write(source_code)
    try:
        subprocess.run(
            ["2to3", filename, "-w"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Read the modified content from the file
        with open(filename, "r") as file:
            modified_code = file.read()
        return modified_code
    except subprocess.CalledProcessError as e:
        # TODO: see if there's any other way to solve this issues...
        raise RuntimeError(f"Error during 2to3 conversion: {e.stderr.decode()}") from e
    finally:
        # remove the helper file
        if os.path.exists(filename):
            os.remove(filename)


def apply_AST_transform_and_write(source: str, ast_transformer) -> str:
    """
    Apply the ast.NodeTransformer class on the source code and return the transformed code
    """
    # parse the source code into AST
    # print('source code: ', source)

    try:
        # If Python 3.x syntax, then no error will be raised
        original_ast_module = ast.parse(source)
    except SyntaxError:
        # Ohterwise, convert to pyhon 3.x syntax from python 2.x
        fixed_source = convert_python2_to_python3(source)
        original_ast_module = ast.parse(fixed_source)
    # rewrite on the AST
    modified_ast_module = ast_transformer.visit(original_ast_module)
    # turn the AST back to code
    modified_code = ast.unparse(modified_ast_module)

    return modified_code


def transform_and_write(
    code_transformer, source_file: str, target_directory: str
) -> None:
    """
    Process each line of the jsonl file,
    """
    # Extract the base name of the source file and append '_transformed'
    source_basename = os.path.basename(source_file)  # Get the file name with extension
    target_filename = os.path.join(
        target_directory,
        source_basename.replace(
            ".jsonl", f"_transformed_{code_transformer.method}.jsonl"
        ),
    )

    with jsonlines.open(source_file) as reader, jsonlines.open(
        target_filename, mode="w"
    ) as writer:
        i = 0
        for obj in reader:

            try:
                transformed_code = apply_AST_transform_and_write(
                    obj["code"], code_transformer
                )
            except RuntimeError:
                # if the code itself has syntax error, then skip it
                continue
            except SyntaxError:
                # if the code itself has syntax error, then skip it
                continue

            output_obj: CodeSearchNetExample = CodeSearchNetExample(
                repo=obj["repo"],
                func_name=obj["func_name"],
                language=obj["language"],
                code=obj["code"],
                docstring=obj["docstring"],
                transformed=transformed_code,
                aug_type=code_transformer.augtype,
            )

            # add this jsonline to target_filename
            writer.write(output_obj.__dict__)
            i += 1

    print(f"{i} out of 30000 functions are successfully transformed...")


def main(argv=None):
    import argparse

    arg_parser = argparse.ArgumentParser()
    # Parsing Arguments
    arg_parser.add_argument("ruleId", help="The id of transformation method", type=int)
    arg_parser.add_argument("root", help="The path to jsonl file", type=str)
    arg_parser.add_argument(
        "target",
        help="The target directory where the transformed code are located",
        type=str,
    )
    args = arg_parser.parse_args(argv)

    # Get Code Transformer Given the RuleID
    try:
        code_transformer = code_transform_map[args.ruleId]()
    except KeyError:
        raise ValueError("ruleId does not exist!")

    # Create Output Directory if Does Not Exist
    if not os.path.exists(args.target):
        os.makedirs(args.target)

    print(
        "-------- Selected Transforming Method: ",
        code_transformer.method,
        " -------- \n",
    )

    # usage: python3.11 transform.py 0 python_jsonl/python_train_0.jsonl output/
    if os.path.isdir(args.root):
        raise ValueError("Input should be a file not a directory")
    elif not args.root.endswith(".jsonl"):
        raise ValueError("Input file should be a .jsonl file")
    else:
        # we now only working on "LocalVariableRenaming, ReverseIfElse, and OpAssignment2EqualAssignment"
        transform_and_write(
            code_transformer=code_transformer,
            source_file=args.root,
            target_directory=args.target,
        )

    print("\nFinished Transformed!\n\n")


# python3.11 transform.py 2  dataset/python_train_1.jsonl  output/
if __name__ == "__main__":
    main()
