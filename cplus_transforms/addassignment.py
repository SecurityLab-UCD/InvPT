from clang.cindex import Index, CursorKind
from collections import deque
import random
import sys

def extract_source_code(node):
    """Extract the source code for the node."""
    extent = node.extent
    with open(extent.start.file.name, 'r') as f:
        lines = f.readlines()
    start_line, start_col = extent.start.line, extent.start.column
    end_line, end_col = extent.end.line, extent.end.column
    code = ''.join(lines[start_line-1:end_line])
    return code[start_col-1:end_col-1]

def generalize_function(root_node, source_file, source_code_lines, modifications):
    # Collect all nodes that have function names
    i = 0
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
        if source_file not in str(curr_visit.location):
            continue
        if "+=" in extract_source_code(curr_visit):
            change_nodes.append(curr_visit)

    # Edit all the function names
    for node in change_nodes:
        print(f"Function {node.spelling} will be renamed")
        line_number = node.location.line
        column_number = node.location.column
        statement_name = extract_source_code(node)

        # edited = function_name.replace("++", "+=1")
        index = statement_name.index("+=")
        variable_name = statement_name[:index].strip()
        number = statement_name[index+2:].strip()
        edited = f"{variable_name}={variable_name}+{number}"

        # Update the source line to replace the function name
        line = source_code_lines[line_number - 1]
        modified_line = line[:column_number - 1] + edited + line[column_number + len(statement_name) - 1:]
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
    generalize_function(tu.cursor, source_file, source_code_lines, modifications)

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

