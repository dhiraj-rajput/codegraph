"""
Symbol extractor — transforms parsed files into a unified symbol table.

Each code entity (function, class, method, variable, import) becomes a
CodeSymbol with a unique identifier, making it easy to index and retrieve.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from parser.tree_sitter_parser import ParsedFile, FunctionInfo, ClassInfo


@dataclass
class CodeSymbol:
    """
    A single code entity extracted from a source file.

    This is the fundamental unit of the retrieval system — each symbol
    becomes a searchable record in BM25, symbol index, and vector index.
    """
    symbol_id: str                   # Unique ID (hash of file + name + line)
    name: str
    qualified_name: str              # e.g. "ClassName.method_name"
    type: str                        # function | class | method | variable | import
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent: Optional[str] = None     # Class name for methods
    calls: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    type_hints: List[str] = field(default_factory=list)
    language: str = "python"

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(self.source_code) // 4


def _make_id(file_path: str, name: str, line: int) -> str:
    """Generate a deterministic unique ID for a symbol."""
    raw = f"{file_path}::{name}::{line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SymbolExtractor:
    """
    Converts ParsedFile objects into flat lists of CodeSymbol records.

    This normalizes the heterogeneous AST output into a uniform symbol table
    that downstream indexes (BM25, symbol index, page index) can consume.
    """

    def extract_from_file(self, parsed: ParsedFile) -> List[CodeSymbol]:
        """Extract all symbols from a single parsed file."""
        symbols = []

        # Functions (module-level)
        for func in parsed.functions:
            sym = self._function_to_symbol(func, parsed.language)
            symbols.append(sym)

        # Classes and their methods
        for cls in parsed.classes:
            cls_sym = self._class_to_symbol(cls, parsed.language)
            symbols.append(cls_sym)

            for method in cls.methods:
                method_sym = self._function_to_symbol(
                    method, parsed.language, parent_class=cls.name
                )
                symbols.append(method_sym)

        # Imports (as symbols for dependency tracking)
        for imp in parsed.imports:
            imp_sym = CodeSymbol(
                symbol_id=_make_id(imp.file_path, imp.module, imp.line_number),
                name=imp.module,
                qualified_name=imp.module,
                type="import",
                file_path=imp.file_path,
                line_start=imp.line_number,
                line_end=imp.line_number,
                source_code=f"import {imp.module}" + (
                    f" ({', '.join(imp.names)})" if imp.names else ""
                ),
                language=parsed.language,
            )
            symbols.append(imp_sym)

        # Variables
        for var in parsed.variables:
            var_sym = CodeSymbol(
                symbol_id=_make_id(var.file_path, var.name, var.line_number),
                name=var.name,
                qualified_name=var.name,
                type="variable",
                file_path=var.file_path,
                line_start=var.line_number,
                line_end=var.line_number,
                source_code=f"{var.name} = {var.value_preview or '...'}",
                language=parsed.language,
            )
            symbols.append(var_sym)

        return symbols

    def extract_from_repository(self, parsed_files: List[ParsedFile]) -> List[CodeSymbol]:
        """Extract all symbols from a list of parsed files."""
        all_symbols = []
        for pf in parsed_files:
            all_symbols.extend(self.extract_from_file(pf))
        return all_symbols

    # ── Internal ─────────────────────────────────────────────────────────

    def _function_to_symbol(
        self, func: FunctionInfo, language: str, parent_class: Optional[str] = None
    ) -> CodeSymbol:
        """Convert FunctionInfo to CodeSymbol."""
        parent = parent_class or func.parent_class
        qualified = f"{parent}.{func.name}" if parent else func.name
        sym_type = "method" if parent else "function"

        # Build signature
        params_str = ", ".join(func.parameters)
        ret = f" -> {func.return_type}" if func.return_type else ""
        signature = f"def {func.name}({params_str}){ret}"

        return CodeSymbol(
            symbol_id=_make_id(func.file_path, qualified, func.line_start),
            name=func.name,
            qualified_name=qualified,
            type=sym_type,
            file_path=func.file_path,
            line_start=func.line_start,
            line_end=func.line_end,
            source_code=func.source_code,
            docstring=func.docstring,
            signature=signature,
            parent=parent,
            calls=func.calls,
            decorators=func.decorators,
            type_hints=func.type_hints,
            language=language,
        )

    def _class_to_symbol(self, cls: ClassInfo, language: str) -> CodeSymbol:
        """Convert ClassInfo to CodeSymbol."""
        bases_str = f"({', '.join(cls.base_classes)})" if cls.base_classes else ""
        signature = f"class {cls.name}{bases_str}"

        return CodeSymbol(
            symbol_id=_make_id(cls.file_path, cls.name, cls.line_start),
            name=cls.name,
            qualified_name=cls.name,
            type="class",
            file_path=cls.file_path,
            line_start=cls.line_start,
            line_end=cls.line_end,
            source_code=cls.source_code,
            docstring=cls.docstring,
            signature=signature,
            base_classes=cls.base_classes,
            decorators=cls.decorators,
            language=language,
        )
