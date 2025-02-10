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

def generate_reverse_code(if_node):
    # Base Case
    if if_node.kind != CursorKind.IF_STMT:
        expr_node = extract_source_code(if_node)
        expr = expr_node[expr_node.index("{")+1:expr_node.index("}")]
        return expr
    children = list(if_node.get_children())
    condition_node = extract_source_code(children[0])
    highest_index = len(condition_node) - 1 - condition_node[::-1].index("{")
    condition = condition_node[:highest_index].strip()
    expr_one = generate_reverse_code(children[1])
    expr_two = "\n\n"
    if len(children) > 2:
        expr_two =  generate_reverse_code(children[2])
    return "if (!%s) {%s} else {%s}" % (condition, expr_two, expr_one)

def if_else_reverse(root_node, source_file, source_code_lines, modifications):
    # Collect all nodes that have IF_STMT
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        if curr_visit.kind == CursorKind.IF_STMT and source_file in str(curr_visit.location):
            changed_code = generate_reverse_code(curr_visit)
            print(changed_code)
            change_nodes.append((curr_visit, changed_code))
            continue
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
    
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
        file_code = file_code[:lines[line_number - 1] + column_number - 1] + hidden_name + file_code[lines[line_number - 1] + len(hidden_name) + column_number - 2:]
        print(file_code)


    length_sorted = list(replace_dictionary.keys())
    length_sorted.sort(reverse=True,key=len)
    # Replace placeholder names with actual names
    for key in length_sorted:
        file_code = file_code.replace(key, replace_dictionary[key])

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
    code = if_else_reverse(tu.cursor, source_file, source_code_lines, modifications)

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

