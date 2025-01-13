import ast

class ReverseIfElser(ast.NodeTransformer):

    def visit_If(self, node):
        self.generic_visit(node)
        
        # Step 1: Negating the condition.
        negated_condition = ast.UnaryOp(op=ast.Not(), operand=node.test)
        # Step 2: Swap the `then` and `else` branches
        new_then = node.orelse or []
        new_else = node.body 
        
        new_if = ast.If(
            test=negated_condition,
            body=new_then,
            orelse=new_else
        )

        return ast.fix_missing_locations(new_if)

def reverse_if_else(source_code):
    tree = ast.parse(source_code)    
    transformer = ReverseIfElser()
    transformed_tree = transformer.visit(tree) 
    return ast.unparse(transformed_tree)

# Example Usage
source_code = """
if a > 5:
    print("a is greater than 5")
else:
    print("a is 5 or less")


if (a < 5):
    print("then code")
elif (a < 10):
    print("elif code")
else:  
    print ( 'else code') 
"""

transformed_code = reverse_if_else(source_code)
print("Original Code:")
print(source_code)
print("\nTransformed Code:")
print(transformed_code)
