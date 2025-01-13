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
        

def transform_for_loops_to_while(source_code: str) -> str:
    """
    Parse the given source_code, transform all for loops to while loops,
    and return the modified code as a string.
    """
    tree = ast.parse(source_code)
    transformer = WhileToForTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)

# --------------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    code_ranges = r'''
x = 1
y = 100
while x < y:
    x *= 2
    y -= 5
    print("x:", x, ", y:", y)
    '''    

#     garbage_code = r'''
# x = 1
# y = 100
# while x < y:
#     x *= 2
#     y -= 5
#     print("x:", x, ", y:", y)
# for _ in range(10000):
#     if not (x < y):
#         break
#     x *= 2
#     y -= 5
#     print('hellow world')
# '''

    new_code = transform_for_loops_to_while(code_ranges)
    print("Original Code:")
    print(code_ranges)
    print("Transformed Code:")
    print(new_code)

    # print(ast.dump(ast.parse(garbage_code), indent=4))