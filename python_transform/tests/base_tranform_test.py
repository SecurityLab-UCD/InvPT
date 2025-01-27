import unittest
import ast

class baseCodeTransformTest(unittest.TestCase):
    def get_transformed_code(self, code: str, code_transform_class) -> str:
        """
        Return the transformed code snippet based on a specific code transform rule
        """
        tree = ast.parse(code)
        transformer = code_transform_class()
        transformed_tree = transformer.visit(tree)
        return ast.unparse(transformed_tree)

    def assert_code_equal(self, code1: str, code2: str):
        """
        Compares two codes in AST format.
        
        Helper Functons from the AST module
        ast.parse(): Parse the source code into AST node
        ast.dump(): Return a formatted dump of the tree in node.         
        """
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        self.assertEqual(ast.dump(tree1), ast.dump(tree2))
