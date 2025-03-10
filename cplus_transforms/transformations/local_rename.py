from clang.cindex import Index, CursorKind
import clang
from collections import deque
import random
import sys
from cplus_transforms.transformations.ast_util import *
    
generated_names = set()

clang.cindex.Config.set_library_file('/usr/lib/llvm-15/lib/libclang.so.1')

def map_num_char(i):
    i = i % 53
    if i == 0:
        return '_'
    elif i <= 26:
        return chr(i + ord('a') - 1)
    else:
        return chr(i + ord('A') - 27)

def generate_random_name(seed = 2023):
    global generated_names
    generator = random.Random()
    generator.seed(seed)
    length = generator.randint(3, 27)
    random_name = ""
    while random_name in generated_names or random_name == "":
        random_name = ""
        for i in range(length):
            random_name += map_num_char(generator.randint(0, 52))
    generated_names.add(random_name)
    return random_name

def local_renamer(root_node, file_code):
    return local_rename(root_node, "example.cpp", file_code, [0, len(file_code)])

def local_rename(root_node, source_file, file_code, start_end):
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
        offset = get_character_offset(file_code, node.location.line, node.location.column)
        hidden_name = function_name_changes[node.spelling]
        file_code = file_code[:offset] + hidden_name + file_code[offset + len(hidden_name):]
        print(file_code)

    length_sorted = list(function_actual_name.keys())
    length_sorted.sort(reverse=True,key=len)
    # Replace placeholder names with actual names
    for key in length_sorted:
        file_code = file_code.replace(key, function_actual_name[key])

    return file_code

def main(code):
    index = clang.cindex.Index.create()
    translation_unit = index.parse('example.cpp', unsaved_files=[('example.cpp', code)], options=0)
    print(local_renamer(translation_unit.cursor, code))

if __name__ == "__main__":
    code = code = """
#include <stdio.h>

int main() {
    int x = 0;
    x = 10;
    printf("Hello, World!\\n");
    return 0;
}
"""
    main(code)

