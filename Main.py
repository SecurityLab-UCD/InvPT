from typing import *
from Methods.LocalVariableRenaming import LocalVariableRenamer
import ast
import sys
import os

transformed_method_map = {
    0: 'Local Variable Renaming',
    1: 'Function Definition Reorder',
}

def get_AST_transformer(ruleId: str):
    if ruleId == 0: # Local Variable Renaming
        return LocalVariableRenamer()
    elif ruleId == 1: # Function Definition Renaming
        return None
    else:
        raise ValueError('ruleId does not exist')
    
def apply_AST_transform_and_write(ast_transformer, source_filename, target_filename):
    with open(source_filename) as file:
        lines = file.readlines()
        source = ''.join(lines)
        module = ast.parse(source)

        modified_module = ast_transformer.visit(module)

        modified_code = ast.unparse(modified_module)  
        
        with open(target_filename, 'w') as file:
            file.write(modified_code)

def main(argv=None):
    import argparse
 
    arg_parser = argparse.ArgumentParser()    

    # Parsing Arguments
    arg_parser.add_argument('ruleId', help='The id of transformation method', type=int)
    arg_parser.add_argument('root', help='The root directory where all code to be transformed are located', type=str)
    arg_parser.add_argument('target', help='The target directory where the transformed code are located', type=str)        
    args = arg_parser.parse_args(argv)
    
    # Get AST Node Transformer Given the RuleID
    ast_transformer = get_AST_transformer(args.ruleId)    

    # traverse through the source directory/file
    # for each file in the directory, transformed it, and then write the output to the target directory    

    if not os.path.exists(args.target):
        os.makedirs(args.target)

    print('-------- Selected Transforming Method: ', transformed_method_map[args.ruleId], ' -------- \n')
    if os.path.isdir(args.root):
        for dirpath, dirnames, filenames in os.walk(args.root):
            for filename in filenames:                
                source_filename = os.path.join(dirpath, filename)
                target_filename = os.path.join(args.target, filename)
                print(source_filename, target_filename)
                apply_AST_transform_and_write(ast_transformer=ast_transformer, source_filename=source_filename, target_filename=target_filename)
    else:
        apply_AST_transform_and_write(ast_transformer=ast_transformer, source_filename=args.root, target_filename=args.target)

    print('\nFinished Transformed!\n\n')




if __name__ == '__main__':
    main()
    