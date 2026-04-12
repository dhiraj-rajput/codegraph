"""
Edge type definitions for the code graph.
"""

from enum import Enum


class EdgeType(str, Enum):
    """Types of relationships between code entities."""
    CALLS = "CALLS"                  # Function A calls Function B
    IMPORTS = "IMPORTS"              # File A imports from File B
    EXTENDS = "EXTENDS"             # Class A inherits from Class B
    DEFINED_IN = "DEFINED_IN"       # Symbol X is defined in File Y
    USES = "USES"                   # Function A references Variable X
    USES_TYPE = "USES_TYPE"         # Function A takes Argument of Type T
    RETURNS = "RETURNS"             # Function A returns type T
    CONTAINS = "CONTAINS"           # Class contains Method
    OVERRIDES = "OVERRIDES"         # Method overrides parent method
    DECORATES = "DECORATES"         # Decorator applied to function/class


class NodeType(str, Enum):
    """Types of nodes in the code graph."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    FILE = "file"
    MODULE = "module"
    VARIABLE = "variable"
    IMPORT = "import"
