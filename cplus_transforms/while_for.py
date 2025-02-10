from clang.cindex import Index, CursorKind
from collections import deque
import random
import sys

# Traverse and get all if statements
def generate_hidden_name(i, str):
    generator = random.Random()
    generator.seed(i + 2025)
    length = len(str)
    random_name = ""
    for i in range(length):
        name_code = generator.randint(16, 31)
        random_name += chr(name_code)
    return random_name

def extract_source_code(node):
    """Extract the source code for the node."""
    extent = node.extent
    print(extent)
    with open(extent.start.file.name, 'r') as f:
        lines = f.readlines()
    start_line, start_col = extent.start.line, extent.start.column
    end_line, end_col = extent.end.line, extent.end.column
    code = ''.join(lines[start_line-1:end_line])
    length = 0
    for i in range(start_line-1, end_line):
        length += len(lines[i])
    return code[start_col-2:end_col-1+length]

def generate_replace_code(condition_node):
    condition = extract_source_code(condition_node).strip().strip("}{")
    return "(;%s;){" % (condition)

def while_for(root_node, source_file, source_code_lines, modifications):
    # Collect all nodes that have WHILE_STMT
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
        if curr_visit.kind == CursorKind.WHILE_STMT and source_file in str(curr_visit.location):
            condition_node = list(curr_visit.get_children())[0]
            changed_code = generate_replace_code(condition_node)
            print(changed_code)
            change_nodes.append((condition_node, changed_code))
            continue
    
    replace_dictionary = {}
    file_code = "".join(source_code_lines)
    lines = []
    length = 0
    for line in source_code_lines:
        lines.append(length)
        length += len(line)

    i = 0
    # Replace with names that are same length and present
    for change_node in change_nodes:
        line_number = change_node[0].location.line
        column_number = change_node[0].location.column
        source_code = extract_source_code(change_node[0])
        hidden_name = generate_hidden_name(i, source_code)
        replace_dictionary[hidden_name] = change_node[1]
        i += 1
        file_code = file_code[:lines[line_number - 1] + column_number - 2] + hidden_name + file_code[lines[line_number - 1] + len(hidden_name) + column_number - 2:]
        print(file_code)


    length_sorted = list(replace_dictionary.keys())
    length_sorted.sort(reverse=True,key=len)
    # Replace placeholder names with actual names
    for key in length_sorted:
        file_code = file_code.replace(key, replace_dictionary[key])

    # Replace all while's with fors
    file_code = file_code.replace("while ", "for ")
    file_code = file_code.replace("while(", "for(")

    return file_code

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
    code = while_for(tu.cursor, source_file, source_code_lines, modifications)

    # Write the modified source code to the output file
    with open(output_file, "w") as f:
        f.write(code)

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

