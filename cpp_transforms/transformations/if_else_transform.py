from clang.cindex import Index, CursorKind
from collections import deque
import sys
from cpp_transforms.transformations.ast_util import extract_source_code, get_node_char_positions, get_offset, generate_hidden_name
import clang

clang.cindex.Config.set_library_file('/usr/lib/llvm-15/lib/libclang.so.1')

def if_else_reverser(root_node: Index, file_code: str) -> str:
    return if_else_reverse(root_node, "example.cpp", file_code, [0, len(file_code)])

def if_else_reverse(root_node: Index, source_file: str, source_code: str, start_end: tuple[int, int]) -> str:
    # Collect all nodes that have IF_STMT
    change_nodes = []
    to_visit = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        if curr_visit.kind == CursorKind.IF_STMT and source_file in str(curr_visit.location):
            # Behavior here to get the children and stuff
            children = list(curr_visit.get_children())
            condition = extract_source_code(children[0], source_code)
            expr_one = if_else_reverse(children[1], source_file, source_code, get_node_char_positions(children[1], source_code))
            expr_two = ""
            if len(children) > 2:
                expr_two =  if_else_reverse(children[2], source_file, source_code, get_node_char_positions(children[2], source_code))
            # Get changed code
            #print("Full Scope", source_code[start_end[0]:start_end[1]])
            #print("Expr_one: ", expr_one)
            #print("Expr_two: ", expr_two)
            changed_code = "if (!(%s)) {%s} else {%s}" % (condition, expr_two, expr_one)
            #print("Changed code: ", changed_code)
            change_nodes.append((curr_visit, changed_code))
            continue
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
    
    replace_dictionary = {}
    file_code = source_code

    i = 0
    # Replace with names that are same length and present
    for change_node in change_nodes:
        offset = get_offset(change_node[0], source_code)
        source_code = extract_source_code(change_node[0], source_code)
        hidden_name = generate_hidden_name(source_code)
       # print(len(source_code))
        replace_dictionary[hidden_name] = change_node[1]
        i += 1
        file_code = file_code[:offset] + hidden_name + file_code[offset + len(hidden_name):]
       # print("File Code Here 1: ", file_code)

    # Replace with actual scope of return before editing
    file_code = file_code[start_end[0]:start_end[1]]
   # print("File Code Here 2: ", file_code)

    length_sorted = list(replace_dictionary.keys())
    length_sorted.sort(reverse=True,key=len)
    # Replace placeholder names with actual names
    for key in length_sorted:
        file_code = file_code.replace(key, replace_dictionary[key])

    return file_code

def main(code: str) -> None:
    index = clang.cindex.Index.create()
    translation_unit = index.parse('example.cpp', unsaved_files=[('example.cpp', code)], options=0)
    print(if_else_reverser(translation_unit.cursor, code))

if __name__ == "__main__":
    test_code = """
#include <stdio.h>

int main() {
    if(true) {
        i++;
    }
}
"""
    main(test_code)

