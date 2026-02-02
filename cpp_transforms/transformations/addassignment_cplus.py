import clang.cindex
from clang.cindex import Cursor
from collections import deque
from cpp_transforms.transformations.ast_util import (
    extract_source_code,
    NUMBER_TYPES,
    generate_hidden_name,
)
from cpp_transforms.transformations.cursor_util import is_comp_assign, is_decl_stmt
import clang


def add_assignmenter_cplus(root_node: Cursor, file_code: str) -> str:
    source_code_lines = file_code.splitlines(keepends=True)
    add_assignment(root_node, "example.cpp", source_code_lines)
    return "".join(source_code_lines)


def add_assignment(
    root_node: Cursor, source_file: str, source_code_lines: list[str]
) -> None:
    # File code
    file_code = "".join(source_code_lines)
    # Collect all nodes that have function names
    is_valid = set()
    i = 0
    change_nodes = []
    to_visit: deque[Cursor] = deque()
    to_visit.appendleft(root_node)
    while len(to_visit) > 0:
        curr_visit = to_visit.pop()
        for child in curr_visit.get_children():
            to_visit.appendleft(child)
        if source_file not in str(curr_visit.location):
            continue
        if is_decl_stmt(curr_visit):
            source_code = extract_source_code(curr_visit, file_code)
            assignment_index = source_code.index("=")
            arr = source_code[:assignment_index].split()
            type = " ".join(arr[:-1])
            var_name = arr[-1]
            if type in NUMBER_TYPES:
                is_valid.add(var_name)
        if "+=" in extract_source_code(curr_visit, file_code) and is_comp_assign(
            curr_visit
        ):
            change_nodes.append(curr_visit)

    replace_dictionary = {}

    i = 0
    # Replace with names that are same length and present
    for node in change_nodes:
        line_number = node.location.line
        column_number = node.location.column
        structure_name = extract_source_code(node, file_code).strip()

        index = structure_name.index("+=")
        variable_name = structure_name[:index].strip()
        number = structure_name[index + 2 :].strip()

        if variable_name not in is_valid:
            continue

        edited = generate_hidden_name(structure_name)
        i += 1
        replace_dictionary[edited] = f"(({variable_name})=({variable_name})+({number}))"
        # print(f"Expression {structure_name} will be renamed to {replace_dictionary[edited]}")

        # Update the source line to put the placeholder name
        line = source_code_lines[line_number - 1]
        modified_line = (
            line[: column_number - 1]
            + edited
            + line[column_number + len(structure_name) - 1 :]
        )
        source_code_lines[line_number - 1] = modified_line

    # Replace placeholder names with actual names
    for i in range(len(source_code_lines)):
        for key in replace_dictionary.keys():
            source_code_lines[i] = source_code_lines[i].replace(
                key, replace_dictionary[key]
            )


def main(code: str) -> None:
    index = clang.cindex.Index.create()
    translation_unit = index.parse(
        "example.cpp", unsaved_files=[("example.cpp", code)], options=0
    )
    print(add_assignmenter_cplus(translation_unit.cursor, code))


if __name__ == "__main__":
    test_code = """
#include <stdio.h>

int main() {
    int x = 0;
    x = 10;
    x += 1;
    printf("Hello, World!\\n");
    return 0;
}
"""
    main(test_code)
