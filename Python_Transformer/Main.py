from typing import *
from Python_Transformer.methods import op_assignment_to_equal_assignment
from methods import LocalVariableRenamer, FunctionDefinitionReorder, ReverseIfElser, StatementOrderRearrangement, WhileToForTransformer, ForToWhileTransformer
import ast
import sys
import os

transformed_method_map = {
    0: 'Local Variable Renaming',
    1: 'Function Definition Reorder',
    2: 'Reverse If Else Statement',
    3: 'Statements Order Rearrangement',
    4: 'Operation Assignment to EqualAssignment',
    5: 'While to For',
    6: 'For to While',
}

def get_code_transformer(ruleId: str):
    if ruleId == 0: # Local Variable Renaming
        return LocalVariableRenamer()
    elif ruleId == 1: # Function Definition Reorder
        return FunctionDefinitionReorder()
    elif ruleId == 2:
        return ReverseIfElser()
    elif ruleId == 3:
        return StatementOrderRearrangement()
    elif ruleId == 4:
        return op_assignment_to_equal_assignment()
    elif ruleId == 5:
        return WhileToForTransformer()
    elif ruleId == 6:
        return ForToWhileTransformer()
    else:
        raise ValueError('ruleId does not exist')
    
def apply_AST_transform_and_write(ast_transformer, source_filename, target_filename):
    with open(source_filename) as file:
        lines = file.readlines()
        source = ''.join(lines)
        
        original_ast_module = ast.parse(source)
        modified_ast_module = ast_transformer.visit(original_ast_module)
        
        modified_code = ast.unparse(modified_ast_module)          
        with open(target_filename, 'w') as file:
            file.write(modified_code)

def apply_function_def_reorder_and_write(code_transformer, source_filename, target_filename):
    
    """    
    modified_code = code_transformer(source_filename)    
    """
    code_transformer.write(source_filename, target_filename)

def apply_statements_order_rearrangement_and_write(code_transformer, source_filename, target_filename):
    code_transformer.write(source_filename, target_filename)


def transform_and_write(code_transformer, source_filename, target_filename, ruleId):
    """
    Use code transformer to generate 
    """
    if ruleId == 1:
        return apply_function_def_reorder_and_write(code_transformer, source_filename, target_filename)
    elif ruleId == 3:
        return apply_statements_order_rearrangement_and_write(code_transformer, source_filename, target_filename)
    else:
        apply_AST_transform_and_write(ast_transformer=code_transformer, source_filename=source_filename, target_filename=target_filename)


def main(argv=None):
    import argparse
 
    arg_parser = argparse.ArgumentParser()    

    # Parsing Arguments
    arg_parser.add_argument('ruleId', help='The id of transformation method', type=int)
    arg_parser.add_argument('root', help='The root directory where all code to be transformed are located', type=str)
    arg_parser.add_argument('target', help='The target directory where the transformed code are located', type=str)        
    args = arg_parser.parse_args(argv)
    
  
    # Create Output File if Does Not Exist
    if not os.path.exists(args.target):
        os.makedirs(args.target)

    print('-------- Selected Transforming Method: ', transformed_method_map[args.ruleId], ' -------- \n')

    # Get Code Transformer Given the RuleID
    code_transformer = get_code_transformer(args.ruleId)  
    
    if os.path.isdir(args.root):
        for dirpath, dirnames, filenames in os.walk(args.root):
            for filename in filenames:                
                source_filename = os.path.join(dirpath, filename)
                target_filename = os.path.join(args.target, filename)
                print(source_filename, target_filename)
                transform_and_write(code_transformer=code_transformer, source_filename=source_filename, target_filename=target_filename, ruleId=args.ruleId)
    else:        
        transform_and_write(ast_transformer=code_transformer, source_filename=args.root, target_filename=args.target, ruleId=args.ruleId)

    print('\nFinished Transformed!\n\n')

if __name__ == '__main__':
    main()
    