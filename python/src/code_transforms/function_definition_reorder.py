import ast
import random


class FunctionDefinitionReorder:
    def write(self, source_filename, target_filename):
        """
        Do transform and make changes to the target directory
        """
        with open(source_filename) as file:
            lines = file.readlines()
            last_line_number = len(lines)
            source = "".join(lines)
            module = ast.parse(source)  # file as AST.module node

            function_ranges = self.get_function_ranges_at_level(
                module
            )  # function range: [start_line_number, end_line_number]
            code_blocks = self.get_code_blocks(
                function_ranges=function_ranges,
                first_line=1,
                last_line=last_line_number,
            )  # block: [start_line_number, end_line_number, is_function]
            self.reorder_functions_relatively(code_blocks)
            self.write_reordered_file(lines, target_filename, code_blocks)

    def get_function_ranges_at_level(self, parent):
        """
        Get all the functions' range, defined by [start_line_number, end_line_number], under an AST node.

        Args:
            parent (AST): it can be a AST.module, or AST.FunctionDef

        Returns:
            list: A new list of functions' ranges
        """

        function_ranges = []
        # Iterate though each ast children
        for node in parent.body:
            if isinstance(node, ast.FunctionDef):
                function_ranges.append((node.lineno, node.end_lineno))
        return function_ranges

    def get_code_blocks(self, function_ranges, first_line, last_line):
        """
        Get the code block of a file. Each block is defined by [start_line_number, end_line_number, is_function]

        Args:
            function_ranges (list): a list of function ranges
            first_line (int): the first line of the input file. It should be 1.
            last_line (int): the last line of the input file.

        Returns:
            list: A new list of blocks
        """
        current_line = first_line
        blocks = []  # (start, end, is_function)

        for start, end in function_ranges:
            if current_line < start:
                # add non-functional code block
                blocks.append((current_line, start - 1, 0))
            # add the function block
            blocks.append((start, end, 1))
            current_line = end + 1

        if current_line <= last_line:
            blocks.append((current_line, last_line, 0))

        return blocks

    def reorder_functions_relatively(self, blocks):
        """
        Shuffle the function blocks relatively, with non-function blocks unchanged.

        Example:
            blocks = [1, 2, 3, 4, 5]
            function_indices = [0, 2]
            then one possible result would be = [3, 2, 1, 4, 5], where the first and third blocks are shuffled
        """

        functions_indices = []
        blocks_to_shuffle = []
        for i, block in enumerate(blocks):
            # if is_function
            if block[2]:
                functions_indices.append(i)
                blocks_to_shuffle.append(block)

        random.shuffle(blocks_to_shuffle)

        for i, idx in enumerate(functions_indices):
            blocks[idx] = blocks_to_shuffle[i]

    def write_reordered_file(self, current_file_lines, target_file, blocks):
        new_file_code_blocks = []

        for start, end, is_function in blocks:
            new_file_code_blocks.append(current_file_lines[start - 1 : end])

        with open(target_file, "w") as file:
            for block in new_file_code_blocks:
                file.write("".join(block))
