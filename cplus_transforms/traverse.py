import sys
from clang.cindex import Index, CursorKind

# SOMETHING WRONG IN THE GET_CHARACTER_OFFSET + GET_NODE_CHAR_POSITIONS
def get_character_offset(file_path, line, column):
    """Convert line and column into absolute character offset."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    char_pos = sum(len(lines[i]) for i in range(line - 1))  # +1 for newline
    char_pos += column - 1  # Adjust for zero-based index
    return char_pos

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

def extract_source_code(node):
    """Extract the source code for the node."""
    extent = node.extent
    with open(extent.start.file.name, 'r') as f:
        code = f.read()
    start_end = get_node_char_positions(node)
    return code[start_end[0]:start_end[1]]

def traverse_ast(node, source_file, depth=0):
    """Recursively traverse the AST, printing information about each node."""
    indent = '  ' * depth
    if source_file in str(node.location):
        print(f"{indent}- {node.kind.name} {node.spelling or node.displayname} ({extract_source_code(node)})")
    
    # Visit children recursively
    for child in node.get_children():
        traverse_ast(child, source_file, depth + 1)

def main(source_file):
    """Main function to parse and traverse the AST of a given C++ source file."""
    index = Index.create()
    
    # Parse the file
    tu = index.parse(source_file)
    if not tu:
        print("Unable to parse the translation unit. Please check your input.")
        return
    
    print(f"Traversing AST for: {source_file}")
    print("-" * 40)
    
    # Start AST traversal from the root
    traverse_ast(tu.cursor, source_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <source_file>")
        sys.exit(1)
    
    source_file = sys.argv[1]
    main(source_file)
