import ast

class ForToWhileTransformer(ast.NodeTransformer):
    """
    Transforms all 'for' loops to 'while' loops.
    Case 1: for i in range(start, stop, step): becomes a numeric while loop.
    Case 2: for elem in iterable: becomes an iterator-based while loop.
    """
    def __init__(self):
        # NOTE: negative = -1, second_negative = negative
        # TODO: in case the step is a reference and we need their value while parsing AST, we store their values in a map
        self.variables = dict() # {var: value}

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node) # incase nested for loops

        # Case 1: if it's range-for
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':            
            return self._transform_range_for(node)
        # Case 2: looping over an iterable
        else:
            # We have: for var in <iterable>
            return self._transform_iterable_for(node)

    def _transform_range_for(self, node: ast.For) -> ast.AST:
        """
        [FROM]
            for i in range(start, stop, step):
                body
        [TO]
            i = start
            while i < stop:  # or i > stop if step < 0
                body
                i += step        
        # TODO: case like "for _ in range(...)"         
        """
        args = node.iter.args        
        # Defaults
        start = ast.Constant(value=0)           # Default start=0
        stop  = ast.Constant(value=0)           # We'll set it properly below
        step  = ast.Constant(value=1)           # Default step=1

        if len(args) == 1:            
            stop = args[0]
        elif len(args) == 2:            
            start, stop = args
        elif len(args) == 3:            
            start, stop, step = args
        else:
            # More than 3 or less than 1 => unexpected usage; in this case, do nothing
            return node 

        # while loop 之前的 index i
        assign_i = ast.Assign(
            targets=[node.target],
            value=start
        )

        # if step >= 0: condition is i < stop
        # else:         condition is i > stop
        # (step will never be equal to zero, the interpreter won't allow it)           
        condition = ast.Compare(
            left=node.target,
            ops=[ast.Lt()] if self._is_Lt_equivalent_in_while_loop(step) else [ast.Gt()],
            comparators=[stop]
        )

        # body is node.body plus i += step at the end
        # TODO: op直接改为加法，不需要管实际是正还是负
        increment = ast.AugAssign(
            target=ast.Name(id=node.target.id, ctx=ast.Store()),
            op=ast.Add(), 
            value=step
        )

        # We want to append increment to the loop body
        new_body = node.body + [increment]

        # Add the increment to the end of the current body
        while_node = ast.While(
            test=condition,
            body=new_body,
            orelse=node.orelse
        )

        # Final Code:
        # while i < stop:   # or i > stop
        #     ...
        #    node.body
        #    ...
        #     i += step
        return ast.Module(body=[assign_i, while_node], type_ignores=[])

    def _is_Lt_equivalent_in_while_loop(self, step):
        # if 'step' in for loop is > 0, 
        # converted while loop: while(index < stop)
        if isinstance(step, ast.Constant): return step.value
                
        if isinstance(step, ast.UnaryOp):
            # NOTE: ast.UnaryOp also includes Invert and Not classes
            res = 1
            # if negative sign
            if not isinstance(step.op, ast.UAdd): res *= -1
            
            # if operand is a Constant
            if isinstance(step.operand, ast.Constant): 
                print('return case 1...', res, step.operand.value)
                res *= step.operand.value
            # elif operand is a variable
            elif isinstance(step.operand, ast.Name):
                print('return case 2...')
                res *= self.get_variable_value(ast.operand.id)
            
            return res > 0
        
        # TODO: Future work (because most of the time step is not a reference)
        if isinstance(step, ast.Name):
            return self.get_variable_value(step.id) > 0
    
    def get_variable_value(self, variable_id):
        """
        While parsing the AST, we store the values of variables in the map.
        This functioin simply returns the value if that variable.
        """
        if variable_id not in self.variables:
            raise KeyError('The variable name is not stored in the variable map!')
                
        return self.variables[variable_id]
    

    def _transform_iterable_for(self, node: ast.For) -> ast.AST:
        """
        Convert:
            for elem in iterable:
                body
        to:
            _temp_iter = iter(iterable)
            while True:
                try:
                    elem = next(_temp_iter)
                except StopIteration:
                    break
                body
        """
        # We create a temp variable for the iterator
        temp_iter_name = ast.Name(id="_temp_iter", ctx=ast.Store())
        iter_call = ast.Call(func=ast.Name(id="iter", ctx=ast.Load()), args=[node.iter], keywords=[])

        assign_iter = ast.Assign(
            targets=[temp_iter_name],
            value=iter_call
        )

        # Inside the while loop: a try/except block around `elem = next(_temp_iter)`
        # try:
        #     elem = next(_temp_iter)
        # except StopIteration:
        #     break
        assign_next = ast.Assign(
            targets=[node.target],
            value=ast.Call(
                func=ast.Name(id="next", ctx=ast.Load()),
                args=[ast.Name(id="_temp_iter", ctx=ast.Load())],
                keywords=[]
            )
        )
        except_handler = ast.ExceptHandler(
            type=ast.Name(id="StopIteration", ctx=ast.Load()),
            name=None,
            body=[ast.Break()]
        )
        try_block = ast.Try(
            body=[assign_next],
            handlers=[except_handler],
            orelse=[],
            finalbody=[]
        )

        # while True:
        #     try:
        #         elem = next(_temp_iter)
        #     except StopIteration:
        #         break
        #     body
        while_node = ast.While(
            test=ast.Constant(value=True),
            body=[try_block] + node.body,
            orelse=node.orelse
        )

        # So final code is:
        # _temp_iter = iter(<iterable>)
        # while True:
        #     try:
        #         elem = next(_temp_iter)
        #     except StopIteration:
        #         break
        #     body
        return ast.Module(body=[assign_iter, while_node], type_ignores=[])

def transform_for_loops_to_while(source_code: str) -> str:
    """
    Parse the given source_code, transform all for loops to while loops,
    and return the modified code as a string.
    """
    tree = ast.parse(source_code)
    transformer = ForToWhileTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)

# --------------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    code_ranges = r'''
nums = [3, 4, 5]
negative_one = -1
for x in range(2, 5, 1):
    print(x)

for k in range(3, 5):
    print(x)

for j in range(5, 2, -1):
    for k in range(0, len(nums)):
        print(i, k)
    '''    

#     code_iterables = r'''
# nums = [3, 4, 5]

# for i, val in enumerate(nums):
#     print(i, val)

# for element in ("apple", "banana", "cherry"):
#     print(element)
#     '''

    new_code = transform_for_loops_to_while(code_ranges)
    print("Original Code:")
    print(code_ranges)
    print("Transformed Code:")
    print(new_code)

    # print(ast.dump(ast.parse(code_ranges), indent=4))