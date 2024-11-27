from typing import *
from LocalVariableRenaming import LocalVariableRenamer
import sys

def getASTTransformer(ruleId: str):
    if ruleId == 0: # Local Variable Renaming
        return LocalVariableRenamer()
    elif ruleId == 1: # Function Definition Renaming
        return None
    else:
        raise ValueError('ruleId does not exist')

def main(argv=None):
    import argparse
    
    arg_parser = argparse.ArgumentParser()    

    # Parsing Arguments
    arg_parser.add_argument('ruleId', help='The id of transformation method', type=int)
    arg_parser.add_argument('root', help='The root directory where all code to be transformed are located', type=str)
    arg_parser.add_argument('target', help='The target directory where the transformed code are located', type=str)        
    args = arg_parser.parse_args(argv)
    
    # Get AST Node Transformer Given the RuleID
    ast_transformer = getASTTransformer(args.ruleId)
    print(ast_transformer)

if __name__ == '__main__':
    main()
    