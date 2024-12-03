import ast
import random

class ReverseIfElser(ast.NodeTransformer):
    def __init__(self):
        self.ifs = []  # Store original if-else statements

    def visit_If(self, node):
        """
        Algorithm:
            if orelse is not empty and the orelse part is not one single if statement
            then switch the order with that
        """               
        self.generic_visit(node)
        
        if node.orelse and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):            
            node.body, node.orelse = node.orelse, node.body
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)

        return node
