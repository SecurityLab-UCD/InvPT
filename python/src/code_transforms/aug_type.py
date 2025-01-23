from enum import Enum

class AugType(str, Enum):
    LOCALVARRENAMING = "LocalVarRenaming"
    FOR2WHILE = "For2While"
    WHILE2FOR = "While2For"
    PP2ADDASSIGNMENT = "PP2AddAssignment"
    ADDASSIGNEMNT2EQUALASSIGNMENT = "AddAssignemnt2EqualAssignment"
    REVERSEIFELSE = "ReverseIfElse"