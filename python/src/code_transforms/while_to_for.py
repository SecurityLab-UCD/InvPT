import ast

class WhileToForTransformer(ast.NodeTransformer):
    """
    Transforms all 'While' loops to 'For' loops.
    Step 1: Create an 'infinite' for loop
    Step 2: add a simple if statement with a break in it
    Step 3: the rest of the body code should be exactly the same
    """    
    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)

        if_break = ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=node.test,
            ),
            body=[ast.Break()]
        )

        # Converting the current while loop to the following for loop
        return ast.For(
            target=ast.Name(id='_', ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id='range', ctx=ast.Load()),
                args=[ast.Constant(value=10000)],
            ),
            body=[
                if_break,
                node.body
            ]
        )