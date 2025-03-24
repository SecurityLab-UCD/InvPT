import clang.cindex
from clang.cindex import Index, CursorKind, Cursor
from collections import deque
import sys
from cpp_transforms.transformations.ast_util import (
    extract_source_code,
    generate_hidden_name,
)
from cpp_transforms.transformations.cursor_util import (
    is_comp_assign,
    is_output_using_expr,
    is_unary_op,
    is_decl_stmt,
)


def replace_short_adder(root_node: Cursor, file_code: str) -> str:
    source_code_lines = file_code.splitlines(keepends=True)
    replace_short_add(root_node, "example.cpp", source_code_lines)
    return "".join(source_code_lines)


def replace_short_add(
    root_node: Cursor, source_file: str, source_code_lines: list[str]
) -> None:
    # File code
    file_code = "".join(source_code_lines)
    # Collect all nodes that have function names
    change_nodes = []
    to_visit: deque[tuple[Cursor, Cursor | None]] = deque()
    to_visit.appendleft((root_node, None))
    while len(to_visit) > 0:
        curr_visit, parent = to_visit.pop()
        for child in curr_visit.get_children():
            to_visit.appendleft((child, curr_visit))
        if source_file not in str(curr_visit.location):
            continue
        source_code = extract_source_code(curr_visit, file_code)
        length = len(source_code)
        if is_unary_op(curr_visit):
            if "++" in source_code.strip()[:2]:
                change_nodes.append(curr_visit)
            if "++" in source_code.strip()[length - 2 :] and (
                not is_output_using_expr(parent)
            ):
                change_nodes.append(curr_visit)

    replace_dictionary = {}

    # Replace with names that are same length and present
    for node in change_nodes:
        line_number = node.location.line
        column_number = node.location.column
        structure_name = extract_source_code(node, file_code).strip()

        edited = generate_hidden_name(structure_name)
        replace_dictionary[edited] = "(" + structure_name.strip("+").strip() + "+=1)"
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
    print(replace_short_adder(translation_unit.cursor, code))


if __name__ == "__main__":
    test_code = """
#include <stdio.h>

int main() {
    int x = 0;
    x++;
}
"""
    main(test_code)
