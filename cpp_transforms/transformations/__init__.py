import os
import clang
import clang.cindex


from .addassignment import add_assignmenter
from .local_rename import local_renamer
from .p2add import replace_short_adder
from .for_while import for_while_reverser
from .while_for import while_for_reverser
from .if_else_transform import if_else_reverser
from .addassignment_cplus import add_assignmenter_cplus
from .p2add_cplus import replace_short_adder_cplus

LIBCLANG_PATH = os.environ.get("LIBCLANG_PATH")
assert LIBCLANG_PATH is not None, "Please set the LIBCLANG_PATH environment variable"
clang.cindex.Config.set_library_file(LIBCLANG_PATH)

__all__ = [
    "add_assignmenter",
    "local_renamer",
    "replace_short_adder",
    "for_while_reverser",
    "while_for_reverser",
    "if_else_reverser",
    "add_assignmenter_cplus",
    "replace_short_adder_cplus",
]
