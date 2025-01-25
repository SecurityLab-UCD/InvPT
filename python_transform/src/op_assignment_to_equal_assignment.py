import sys

sys.path.append("../..")
from modeling.dataloader import AugType
import ast
import random
import string


class OpAssignment2EqualAssignment(ast.NodeTransformer):
    """
    For example, it transforms all x += 1 to x = x + 1
    """

    def __init__(self):
        self.augtype: AugType = AugType.ADDASSIGNMENT2EQUALASSIGNMENT
        self.method: str = self.augtype.value

    def visit_AugAssign(self, node: ast.AugAssign):
        new_node = ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=node.target, op=node.op, right=node.value),
        )
        "copy_location: Copy source location (lineno, col_offset, end_lineno, and end_col_offset)"
        "from old_node to new_node if possible, and return new_node"
        return ast.copy_location(new_node, node)
