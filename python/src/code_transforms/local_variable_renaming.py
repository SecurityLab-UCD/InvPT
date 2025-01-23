from typing import Dict
from code_transforms.aug_type import AugType
import ast
import random
import string

class LocalVariableRenamer(ast.NodeTransformer):
    """
    Replace each variable name to a random name.
    A random name is 8 random ascii letters
    """
    def __init__(self):
        # map original variable name to a new random name
        self.augtype: AugType = AugType.LOCALVARRENAMING
        self.method: str = self.augtype.value
        self.variable_mapping : Dict[str:str] = {}

    def generate_random_name(self):        
        return ''.join(random.choices(string.ascii_letters, k=8))
    
    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):        
        self.variable_mapping = {}
        
        # randomnize the function arguments
        for arg in node.args.args:
            new_name = self.generate_random_name()
            self.variable_mapping[arg.arg] = new_name
            arg.arg = new_name
        
        # do the same logic inside the function
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Process variable names
        if isinstance(node.ctx, ast.Store):  # Variable declaration/assignment
            if node.id not in self.variable_mapping:
                self.variable_mapping[node.id] = self.generate_random_name()
            node.id = self.variable_mapping[node.id]
        elif isinstance(node.ctx, ast.Load):  # Variable usage
            if node.id in self.variable_mapping:
                node.id = self.variable_mapping[node.id]
        return node