from transform import TRANSFORMATION_MAP, convert_python2_to_python3, convert_source_to_ast_module
from returns.maybe import Maybe, Nothing, Some
import ast

def get_transformed_codes(source_code: str) -> tuple[str, str, str]:
    """
    Apply three ast.NodeTransformers on the source code, and return three transformed codes.
    The code without any transformations won't be returned
    Returns `Nothing` if all transformations fail, otherwise returns a tuple of three.
    """
    def unparse_ast(transformer: ast.NodeTransformer | None = None, original_code: str = "") -> Maybe[str]:
        maybe_module: Maybe[ast.Module] = convert_source_to_ast_module(
            source_code
        )
        if maybe_module == Nothing:
            return Nothing
        try:
            if transformer is None:
                return maybe_module.map(ast.unparse)
            else:
                transformed_code = maybe_module.map(transformer.visit).map(ast.unparse).value_or(None)                
                return Some(transformed_code) if ( original_code is not None and transformed_code != original_code ) else Nothing                             
        except RecursionError:
            return Nothing

    # Apply all transformations
    original_unparse_result = unparse_ast()
    original_code = original_unparse_result.value_or("")
    transformed_results: List[Maybe[str]] = [
        unparse_ast(TRANSFORMATION_MAP[k](), original_code) for k in TRANSFORMATION_MAP.keys()
    ]    
    # If all transformations fail, return Nothing
    if all(result == Nothing for result in transformed_results):
        return []
    # If some of results are None (when an error is raised while parsing AST tree), mark it as None 
    return tuple(result.value_or(None) for result in transformed_results if result != Nothing)