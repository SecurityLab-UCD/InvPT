import ast

class ReverseIfElseTransformer(ast.NodeTransformer):
    
    def visit_If(self, node):
        """        
        Step 1. Negating the condition.
        Step 2. Swapping the `then` and `else` blocks.
        """
        self.generic_visit(node)  # Visit child nodes if needed

        # Negate the condition
        negated_condition = ast.UnaryOp(op=ast.Not(), operand=node.test)

        # Swap the `then` and `else` branches
        new_then = node.orelse or []  # `else` block becomes the `then` block
        new_else = node.body          # Original `then` block becomes the `else` block
        
        new_if = ast.If(
            test=negated_condition,
            body=new_then,
            orelse=new_else
        )

        # Ensure proper locations in the AST
        return ast.fix_missing_locations(new_if)

def reverse_if_else(source_code):
    tree = ast.parse(source_code)    
    transformer = ReverseIfElseTransformer()
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
