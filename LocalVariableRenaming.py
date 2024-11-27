import ast
import random
import string

"""

"""
class LocalVariableRenamer(ast.NodeTransformer):
    def __init__(self):
        self.variable_mapping = {}

    def generate_random_name(self):
        """Generate a random variable name."""
        return ''.join(random.choices(string.ascii_letters, k=8))

    def visit_FunctionDef(self, node):
        # Reset variable mapping for each function to handle local variables only
        self.variable_mapping = {}
        # Visit function arguments and rename them
        for arg in node.args.args:
            new_name = self.generate_random_name()
            self.variable_mapping[arg.arg] = new_name
            arg.arg = new_name
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

    def visit_Assign(self, node):
        # Handle assignment targets
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node):
        # Handle augmented assignments (e.g., +=, -=)
        self.generic_visit(node)
        return node
