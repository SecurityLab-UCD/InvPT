import ast
import random

class StatementOrderRearrangement(ast.NodeTransformer):
    """
    For each block, swap the order of unrelated statements
    """
    def __init__(self):
        self.unrelated_statement_pairs = []
        self.involved_statements = []



    def visit_Module(self, node: ast.Module):
        
        return node


"""
Conditions of Independent:
1. Two statements do not share common variables
2. avoid function calls for now
3. skip continue, break, and return statement

Plan A (manually swap two unrelated consecutive statemetns):
For Module
    go through each consecutive statements in the module.body
    skip if one of them in (function call, continue, break, and return)
    resolve all the variables of these two statements; if intersection is not None, then skip
    store these two unrelated statements in a map as tuple, also add them in a seen set

    Iterate through each pair in the map:
        switch the order

Plan B (get all statement that are not related with each other)
    Add the first statement in a set
    for each rest of the statements, check if the current statement is not related with all statements in a set; if so, add the it into the set
    randomly reorder these unrelated statements to obtain an invariant program
"""