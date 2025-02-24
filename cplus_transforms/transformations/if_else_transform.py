from clang.cindex import Index, CursorKind
from collections import deque
import sys
from ast_util import *

def if_else_reverse(root_node, source_file, source_code, start_end):
    # Collect all nodes that have IF_STMT
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        if curr_visit.kind == CursorKind.IF_STMT and source_file in str(curr_visit.location):
            # Behavior here to get the children and stuff
            children = list(curr_visit.get_children())
            condition = extract_source_code(children[0])
            expr_one = if_else_reverse(children[1], source_file, source_code, get_node_char_positions(children[1]))
            expr_two = ""
            if len(children) > 2:
                expr_two =  if_else_reverse(children[2], source_file, source_code, get_node_char_positions(children[2]))
            # Get changed code
            print("Full Scope", source_code[start_end[0]:start_end[1]])
            print("Expr_one: ", expr_one)
            print("Expr_two: ", expr_two)
            changed_code = "if (!(%s)) {%s} else {%s}" % (condition, expr_two, expr_one)
            print("Changed code: ", changed_code)
            change_nodes.append((curr_visit, changed_code))
            continue
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
    
    replace_dictionary = {}
    file_code = source_code

    i = 0
    # Replace with names that are same length and present
    for change_node in change_nodes:
        offset = get_offset(change_node[0])
        source_code = extract_source_code(change_node[0])
        hidden_name = generate_hidden_name(source_code)
        print(len(source_code))
        replace_dictionary[hidden_name] = change_node[1]
        i += 1
        file_code = file_code[:offset] + hidden_name + file_code[offset + len(hidden_name):]
        print("File Code Here 1: ", file_code)

    # Replace with actual scope of return before editing
    file_code = file_code[start_end[0]:start_end[1]]
    print("File Code Here 2: ", file_code)

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
    file_code = "".join(source_code_lines)
    code = if_else_reverse(tu.cursor, source_file, file_code, [0, len(file_code)])

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

