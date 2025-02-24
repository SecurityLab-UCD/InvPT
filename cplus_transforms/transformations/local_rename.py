from clang.cindex import Index, CursorKind
from collections import deque
import random
import sys
from ast_util import *

def generate_random_name(i, length = -1):
    generator = random.Random()
    generator.seed(i + 2025)
    if length == -1:
        length = generator.randint(10, 25)
    random_name = ""
    for i in range(length):
        random_name += chr(generator.randint(0, 25) + ord('A'))
    return random_name

def local_rename(root_node, source_file, source_code_lines, modifications):
    # Collect all nodes that have function names
    i = 0
    function_name_changes = {}
    function_actual_name = {}
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
        if source_file not in str(curr_visit.location):
            continue
        if curr_visit.kind == CursorKind.VAR_DECL:
            if curr_visit.spelling not in function_name_changes:
                temp_name = generate_hidden_name(curr_visit.spelling)
                function_name_changes[curr_visit.spelling] = temp_name
                function_actual_name[temp_name] = generate_random_name(i)
                i += 1
            change_nodes.append(curr_visit)
        if curr_visit.kind == CursorKind.DECL_REF_EXPR:
            if curr_visit.spelling in function_name_changes:
                change_nodes.append(curr_visit)

    # Edit all the function names
    for node in change_nodes:
        print(f"Function / Variable {node.spelling} will be renamed")
        line_number = node.location.line
        column_number = node.location.column
        function_name = node.spelling
        
        # Update the source line to replace the function name
        line = source_code_lines[line_number - 1]
        modified_line = line[:column_number - 1] + function_name_changes[function_name] + line[column_number + len(function_name) - 1:]
        source_code_lines[line_number - 1] = modified_line

        # Store modification details (optional logging or rollback)
        modifications.append((line_number, line, modified_line))

def main(source_file, output_file):
    """Parse the source, modify it, and write updated source to a file."""
    index = Index.create()
    tu = index.parse(source_file)

    # Read source code
    with open(source_file, "r") as f:
        source_code_lines = f.readlines()

    print(f"AST Traversal and Modifications for: {source_file}")
    print("-" * 40)

    # Modifications storage
    modifications = []

    # Extract AST and make modifications
    local_rename(tu.cursor, source_file, source_code_lines, modifications)

    # Write the modified source code to the output file
    with open(output_file, "w") as f:
        f.writelines(source_code_lines)

    print("\nModifications Applied:")
    for line_num, old_line, new_line in modifications:
        print(f"Line {line_num}: {old_line.strip()} -> {new_line.strip()}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source_file> <output_file>")
        sys.exit(1)
    source_file = sys.argv[1]
    output_file = sys.argv[2]
    main(source_file, output_file)

