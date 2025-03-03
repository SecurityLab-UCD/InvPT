from clang.cindex import Index, CursorKind
from collections import deque
import sys
from cplus_transforms.transformations.ast_util import *

def add_assignmenter(root_node, file_code: str):
    modifications = []
    return add_assignment(root_node, "example.cpp", file_code.splitlines(), modifications)

def add_assignment(root_node, source_file, source_code_lines, modifications):
    # Collect all nodes that have function names
    is_valid = set()
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
        if curr_visit.kind == CursorKind.DECL_STMT:
            source_code = extract_source_code(curr_visit)
            assignment_index = source_code.index("=")
            arr = source_code[:assignment_index].split()
            type = " ".join(arr[:-1])
            var_name = arr[-1]
            if type in NUMBER_TYPES:
                is_valid.add(var_name)
        if "+=" in extract_source_code(curr_visit) and curr_visit.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            change_nodes.append(curr_visit)

    replace_dictionary = {}

    i = 0
    # Replace with names that are same length and present
    for node in change_nodes:
        line_number = node.location.line
        column_number = node.location.column
        structure_name = extract_source_code(node).strip()
        
        index = structure_name.index("+=")
        variable_name = structure_name[:index].strip()
        number = structure_name[index+2:].strip()

        if variable_name not in is_valid:
            continue

        edited = generate_hidden_name(structure_name)
        i += 1
        replace_dictionary[edited] = f"(({variable_name})=({variable_name})+({number}))"
        print(f"Expression {structure_name} will be renamed to {replace_dictionary[edited]}")

        # Update the source line to put the placeholder name
        line = source_code_lines[line_number - 1]
        modified_line = line[:column_number - 1] + edited + line[column_number + len(structure_name) - 1:]
        source_code_lines[line_number - 1] = modified_line

        # Store modification details (optional logging or rollback)
        modifications.append((line_number, structure_name))

    # Replace placeholder names with actual names
    for i in range(len(source_code_lines)):
        for key in replace_dictionary.keys():
            source_code_lines[i] = source_code_lines[i].replace(key, replace_dictionary[key])

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
    add_assignment(tu.cursor, source_file, source_code_lines, modifications)

    # Write the modified source code to the output file
    with open(output_file, "w") as f:
        f.writelines(source_code_lines)

    print("\nModifications Applied:")
    for line_num, old_line in modifications:
        print(f"Line {line_num}: {old_line.strip()}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source_file> <output_file>")
        sys.exit(1)
    source_file = sys.argv[1]
    output_file = sys.argv[2]
    main(source_file, output_file)

