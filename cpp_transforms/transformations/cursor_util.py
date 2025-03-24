from clang.cindex import Cursor, CursorKind
from typing import no_type_check


@no_type_check
def is_decl_stmt(node: Cursor) -> bool:
    return node.kind == CursorKind.DECL_STMT


@no_type_check
def is_comp_assign(node: Cursor) -> bool:
    return node.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR


@no_type_check
def is_for(node: Cursor) -> bool:
    return node.kind == CursorKind.FOR_STMT


@no_type_check
def is_while(node: Cursor) -> bool:
    return node.kind == CursorKind.WHILE_STMT


@no_type_check
def is_if(node: Cursor) -> bool:
    return node.kind == CursorKind.IF_STMT


@no_type_check
def is_var_decl(node: Cursor) -> bool:
    return node.kind == CursorKind.VAR_DECL


@no_type_check
def is_decl_ref_stmt(node: Cursor) -> bool:
    return node.kind == CursorKind.DECL_REF_EXPR


@no_type_check
def is_unary_op(node: Cursor) -> bool:
    return node.kind == CursorKind.UNARY_OPERATOR


# These expression types utilize the result of x++ or similar expressions as an argument.
# These need to be removed due to x++ not having the same behavior as x+=1 in what it returns (preincrement vs postincrement)
@no_type_check
def is_output_using_expr(node: Cursor) -> bool:
    OUTPUT_USING_EXPRESSION_TYPES = {
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
        CursorKind.PAREN_EXPR,
    }
    return node.kind in OUTPUT_USING_EXPRESSION_TYPES
