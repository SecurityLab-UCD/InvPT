from src.aug_type import AugType
import ast


class ReverseIfElser(ast.NodeTransformer):
    def __init__(self):
        # map original variable name to a new random name
        self.augtype: AugType = AugType.REVERSEIFELSE
        self.method: str = self.augtype.value

    def visit_If(self, node: ast.If):
        self.generic_visit(node)

        # Step 1: Negating the condition.
        negated_condition = ast.UnaryOp(op=ast.Not(), operand=node.test)
        # Step 2: Swap the `then` and `else` branches
        new_then = node.orelse or ast.Expr(value=ast.Constant(value=Ellipsis))
        new_else = node.body

        new_if = ast.If(test=negated_condition, body=new_then, orelse=new_else)

        return ast.fix_missing_locations(new_if)
