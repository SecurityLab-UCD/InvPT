import ast
import random
import string

class LocalVariableRenamer(ast.NodeTransformer):
    def __init__(self):
        # original variable 2 random name
        self.variable_mapping = {}

    def generate_random_name(self):        
        return ''.join(random.choices(string.ascii_letters, k=8))
    
    def visit_Assign(self, node):
        # a = c + b
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node):
        # a += b
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):        
        self.variable_mapping = {}
        
        # randomnize the function arguments
        for arg in node.args.args:
            new_name = self.generate_random_name()
            self.variable_mapping[arg.arg] = new_name
            arg.arg = new_name
        
        # do the same logic inside the function
        return self.generic_visit(node)

    def visit_Name(self, node):
        # Process variable names
        if isinstance(node.ctx, ast.Store):  # Variable declaration/assignment
            if node.id not in self.variable_mapping:
                self.variable_mapping[node.id] = self.generate_random_name()
            node.id = self.variable_mapping[node.id]
        elif isinstance(node.ctx, ast.Load):  # Variable usage
            if node.id in self.variable_mapping:
                node.id = self.variable_mapping[node.id]
        return node