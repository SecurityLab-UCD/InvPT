from clang.cindex import Index, CursorKind
from collections import deque
import sys
import random

EXPRESSION_TYPES = [
    CursorKind.UNEXPOSED_EXPR,
    CursorKind.BINARY_OPERATOR,
    CursorKind.UNARY_OPERATOR,
    CursorKind.CONDITIONAL_OPERATOR,
    CursorKind.CSTYLE_CAST_EXPR,
    CursorKind.COMPOUND_LITERAL_EXPR,
    CursorKind.INIT_LIST_EXPR,
    CursorKind.ADDR_LABEL_EXPR,
    CursorKind.OBJC_STRING_LITERAL,
    CursorKind.OBJC_ENCODE_EXPR,
    CursorKind.OBJC_SELECTOR_EXPR,
    CursorKind.OBJC_PROTOCOL_EXPR,
    CursorKind.OBJC_BRIDGE_CAST_EXPR,
    CursorKind.PACK_EXPANSION_EXPR,
    CursorKind.SIZE_OF_PACK_EXPR,
    CursorKind.PAREN_EXPR
]

def generate_hidden_name(i, length = -1):
    generator = random.Random()
    generator.seed(i + 2025)
    if length == -1:
        length = generator.randint(10, 25)
    name_code = ((11 * i) + 2025) % 32**length
    random_name = ""
    for i in range(length):
        random_name += chr(name_code % 31)
        name_code //= 32
    return random_name

def extract_source_code(node):
    """Extract the source code for the node."""
    extent = node.extent
    with open(extent.start.file.name, 'r') as f:
        lines = f.readlines()
    start_line, start_col = extent.start.line, extent.start.column
    end_line, end_col = extent.end.line, extent.end.column
    code = ''.join(lines[start_line-1:end_line])
    return code[start_col-1:end_col-1]

def replace_short_add(root_node, source_file, source_code_lines, modifications):
    # Collect all nodes that have function names
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft([root_node, None])
    while len(to_visit) > 0:
        [curr_visit, parent] = to_visit.pop()
        for child in curr_visit.get_children():
            to_visit.appendleft([child, curr_visit])
        if source_file not in str(curr_visit.location):
            continue
        source_code = extract_source_code(curr_visit)
        length = len(source_code)
        if curr_visit.kind == CursorKind.UNARY_OPERATOR:
            if "++" in source_code.strip()[:2]:
                 change_nodes.append(curr_visit)
            if "++" in source_code.strip()[length-2:] and not (parent.kind in EXPRESSION_TYPES):
                 change_nodes.append(curr_visit)

    replace_dictionary = {}

    i = 0
    # Replace with names that are same length and present
    for node in change_nodes:
        line_number = node.location.line
        column_number = node.location.column
        structure_name = extract_source_code(node).strip()
        
        edited = generate_hidden_name(i, len(structure_name))
        i += 1
        replace_dictionary[edited] = "(" + structure_name.strip("+").strip() + "+=1)"
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
    replace_short_add(tu.cursor, source_file, source_code_lines, modifications)

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

