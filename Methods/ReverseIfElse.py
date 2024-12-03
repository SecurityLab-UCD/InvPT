import ast
import random

class ReverseIfElser(ast.NodeTransformer):
    def __init__(self):
        self.ifs = []  # Store original if-else statements

    def visit_If(self, node):
        """
        Algorithm:
            For each outmost if statement
            have a copy of this if statment and its following ifelse statement, but without orelse part (the last else statement should be negate)
            randomly reorganize the order of these if statements

            create a new if statement node:
                go through the list of copy nodes, modify the orelse part

            generic visit this new if statemetn node
            return this new if statemetn node
        """                

        # Collect all related if-else blocks, stopping at the final else
        related_ifs = []
        
        current = node
        while isinstance(current, ast.If):
            related_ifs.append(current)
            if current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If): # if it's a elif statement
                current = current.orelse[0]
            else:
                break
        final_else = current.orelse if current.orelse else []

        # Create copies of the nodes without the orelse parts
        copied_ifs = []
        for if_node in related_ifs:         
            copy_node = ast.If(
                test=if_node.test,
                body=if_node.body,
                orelse=[]  
            )
            copied_ifs.append(copy_node)

        # handle the last if-else statement:
        if final_else and len(final_else) == 1 and isinstance(final_else[0], ast.If):
            # if the last one is also an if-else statement
            if_node = final_else[0]
            copy_node = ast.If(test=if_node.test, body=if_node.body, orelse=[])
            copied_ifs.append(copy_node)
        elif final_else:
            # if the last one is a else statement
            last_if = copied_ifs[-1]
            copy_node = ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=last_if.test),
                body=final_else,
                orelse=[]
            )
            copied_ifs.append(copy_node)

        # Randomly shuffle the copied if statements
        random.shuffle(copied_ifs)

        # Negate the condition of the last copied node
        if copied_ifs:
            last_if = copied_ifs[-1]
            last_if.test = ast.UnaryOp(op=ast.Not(), operand=last_if.test)
            last_if.orelse = final_else

        
        for i in range(len(copied_ifs) - 1):
            copied_ifs[i].orelse = [copied_ifs[i + 1]]

        
        new_if_node = copied_ifs[0]
        return new_if_node

    
