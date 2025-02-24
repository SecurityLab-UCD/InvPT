from clang.cindex import CursorKind

# Generates a unique hidden name to be replaced later
def generate_hidden_name(str):
    replaced_char = chr(ord(str[0]) - ord('a') + 1138)
    random_name = replaced_char + str[1:]
    return random_name

# Gives the character offset of a certain line and column position given the file
def get_character_offset(file_path, line, column):
    """Convert line and column into absolute character offset."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    char_pos = sum(len(lines[i]) for i in range(line - 1))  # - 1 Adjust for zero-index
    char_pos += column - 1  # Adjust for zero-based index
    return char_pos

# Gets the starting character offset of a specific AST node
def get_offset(node):
    return get_character_offset(node.location.file.name, node.extent.start.line, node.extent.start.column)

# Function to get start and end character positions of a node
def get_node_char_positions(node):
    """Returns (start_offset, end_offset) of a node."""
    if node.location.file and node.extent.start.file:  # Ensure valid locations
        file_path = node.location.file.name
        # print(node.extent.start)
        start_offset = get_character_offset(file_path, node.extent.start.line, node.extent.start.column)
        end_offset = get_character_offset(file_path, node.extent.end.line, node.extent.end.column)
        return start_offset, end_offset
    return [None, None]

# Gives back the source code for the AST node
def extract_source_code(node):
    """Extract the source code for the node."""
    extent = node.extent
    with open(extent.start.file.name, 'r') as f:
        code = f.read()
    start_end = get_node_char_positions(node)
    return code[start_end[0]:start_end[1]]

# Types that are numerical where += x and = y + x are well-defined
NUMBER_TYPES = [
    "char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "float",
    "double",
    "long double",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
]

# These expression types utilize the result of x++ or similar expressions as an argument. These need to be removed due to x++ not having the same behavior as x+=1 in what it returns (preincrement vs postincrement)
OUTPUT_USING_EXPRESSION_TYPES = [
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