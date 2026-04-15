"""
Tree-sitter based code parser.

Parses source files into structured AST representations, extracting
functions, classes, imports, and other code constructs.
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from tree_sitter import Parser

from parser.languages import get_language, language_for_extension
from config.settings import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Represents a parsed function or method."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None       # Set if this is a method
    calls: List[str] = field(default_factory=list)  # Functions called within
    type_hints: List[str] = field(default_factory=list) # Extracted type annotations


@dataclass
class ClassInfo:
    """Represents a parsed class."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Represents an import statement."""
    module: str
    names: List[str]                          # Imported names (empty for plain import)
    alias: Optional[str] = None
    file_path: str = ""
    line_number: int = 0


@dataclass
class VariableInfo:
    """Represents a module-level variable assignment."""
    name: str
    file_path: str
    line_number: int
    type_hint: Optional[str] = None
    value_preview: Optional[str] = None       # First 100 chars of value


@dataclass
class ParsedFile:
    """Complete parsed representation of a source file."""
    file_path: str
    language: str
    source_code: str
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return self.source_code.count("\n") + 1


# ─── Parser ─────────────────────────────────────────────────────────────────

class CodeParser:
    """
    Multi-language code parser using Tree-sitter.

    Extracts functions, classes, imports, and variables from source files.
    Currently optimized for Python with extensible language support.
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.parser
        self._parsers: Dict[str, Parser] = {}

    def _get_parser(self, language: str) -> Optional[Parser]:
        """Get or create a parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        lang_obj = get_language(language)
        if lang_obj is None:
            return None

        parser = Parser(lang_obj)
        self._parsers[language] = parser
        return parser

    # ── Public API ───────────────────────────────────────────────────────

    def parse_file(self, file_path: str) -> Optional[ParsedFile]:
        """
        Parse a single source file.

        Returns ParsedFile with extracted constructs, or None if
        the file cannot be parsed (unsupported language, too large, etc.).
        """
        path = Path(file_path)

        # Determine language
        lang_name = language_for_extension(path.suffix)
        if lang_name is None:
            return None

        # Check file size
        try:
            if path.stat().st_size > self.config.max_file_size_bytes:
                logger.warning(f"Skipping large file: {file_path}")
                return None
        except OSError:
            return None

        # Read source
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return None

        # Parse
        parser = self._get_parser(lang_name)
        if parser is None:
            logger.debug(f"No parser for language: {lang_name}")
            return None

        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        parsed = ParsedFile(
            file_path=str(path),
            language=lang_name,
            source_code=source,
        )

        # Extract constructs based on language
        if lang_name == "python":
            self._extract_python(root, source, parsed)
        elif lang_name in ("javascript", "typescript"):
            self._extract_javascript(root, source, parsed)
        else:
            # Generic extraction (functions + classes only)
            self._extract_generic(root, source, parsed)

        return parsed

    def parse_repository(self, repo_path: str) -> List[ParsedFile]:
        """
        Walk a repository directory and parse all supported source files.

        Respects ignore patterns from config.
        """
        results = []
        repo = Path(repo_path)

        if not repo.is_dir():
            raise ValueError(f"Not a directory: {repo_path}")

        for root, dirs, files in os.walk(repo):
            # Filter ignored directories
            dirs[:] = [
                d for d in dirs
                if not any(d == p or d.endswith(p) for p in self.config.ignore_patterns)
            ]

            for fname in files:
                fpath = Path(root) / fname
                ext = fpath.suffix

                # Check extension
                if ext not in self.config.supported_extensions:
                    continue

                # Check ignore patterns for files
                if any(fname.endswith(p.lstrip("*")) for p in self.config.ignore_patterns if "*" in p):
                    continue

                parsed = self.parse_file(str(fpath))
                if parsed is not None:
                    results.append(parsed)

        logger.info(f"Parsed {len(results)} files from {repo_path}")
        return results

    # ── Python Extraction ────────────────────────────────────────────────

    def _extract_python(self, root, source: str, parsed: ParsedFile):
        """Extract Python-specific constructs from AST."""
        source_bytes = bytes(source, "utf-8")

        for child in root.children:
            if child.type == "function_definition":
                func = self._extract_python_function(child, source_bytes, parsed.file_path)
                if func:
                    parsed.functions.append(func)

            elif child.type == "decorated_definition":
                # Unwrap decorators
                inner = None
                decorators = []
                for sub in child.children:
                    if sub.type == "decorator":
                        dec_text = sub.text.decode("utf-8").strip().lstrip("@")
                        decorators.append(dec_text)
                    elif sub.type == "function_definition":
                        inner = self._extract_python_function(sub, source_bytes, parsed.file_path)
                        if inner:
                            inner.decorators = decorators
                            parsed.functions.append(inner)
                    elif sub.type == "class_definition":
                        inner = self._extract_python_class(sub, source_bytes, parsed.file_path)
                        if inner:
                            inner.decorators = decorators
                            parsed.classes.append(inner)

            elif child.type == "class_definition":
                cls = self._extract_python_class(child, source_bytes, parsed.file_path)
                if cls:
                    parsed.classes.append(cls)

            elif child.type in ("import_statement", "import_from_statement"):
                imp = self._extract_python_import(child, source_bytes, parsed.file_path)
                if imp:
                    parsed.imports.append(imp)

            elif child.type == "expression_statement":
                var = self._extract_python_variable(child, source_bytes, parsed.file_path)
                if var:
                    parsed.variables.append(var)

    def _extract_python_function(
        self, node, source_bytes: bytes, file_path: str
    ) -> Optional[FunctionInfo]:
        """Extract a Python function definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = name_node.text.decode("utf-8")
        source_code = node.text.decode("utf-8")

        # Extract parameters
        params = []
        type_hints = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for p in params_node.children:
                if p.type in ("identifier", "default_parameter", "list_splat_pattern", "dictionary_splat_pattern"):
                    params.append(p.text.decode("utf-8"))
                elif p.type in ("typed_parameter", "typed_default_parameter"):
                    params.append(p.text.decode("utf-8"))
                    # Extract the type part
                    t_node = p.child_by_field_name("type")
                    if t_node:
                        t_text = t_node.text.decode("utf-8")
                        # Handle Optional[Foo], Union[Foo, Bar] by just grabbing words
                        # re is already imported at module level
                        types = re.findall(r'[a-zA-Z_]\w*', t_text)
                        for t in types:
                            if t not in ("Optional", "Union", "List", "Dict", "Set", "Tuple", "str", "int", "float", "bool", "Any"):
                                type_hints.append(t)

        # Extract return type
        return_type = None
        ret_node = node.child_by_field_name("return_type")
        if ret_node:
            return_type = ret_node.text.decode("utf-8")

        # Extract docstring
        docstring = self._extract_python_docstring(node, source_bytes)

        # Extract function calls
        calls = self._extract_calls(node, source_bytes)

        return FunctionInfo(
            name=name,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_code=source_code,
            docstring=docstring,
            parameters=params,
            return_type=return_type,
            calls=calls,
            type_hints=type_hints,
        )

    def _extract_python_class(
        self, node, source_bytes: bytes, file_path: str
    ) -> Optional[ClassInfo]:
        """Extract a Python class definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = name_node.text.decode("utf-8")
        source_code = node.text.decode("utf-8")

        # Base classes
        bases = []
        superclasses = node.child_by_field_name("superclasses")
        if superclasses:
            for arg in superclasses.children:
                if arg.type == "identifier":
                    bases.append(arg.text.decode("utf-8"))
                elif arg.type == "attribute":
                    bases.append(arg.text.decode("utf-8"))

        # Docstring
        docstring = self._extract_python_docstring(node, source_bytes)

        # Methods
        methods = []
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "function_definition":
                    method = self._extract_python_function(child, source_bytes, file_path)
                    if method:
                        method.parent_class = name
                        methods.append(method)
                elif child.type == "decorated_definition":
                    for sub in child.children:
                        if sub.type == "function_definition":
                            method = self._extract_python_function(sub, source_bytes, file_path)
                            if method:
                                method.parent_class = name
                                decorators = []
                                for dec_node in child.children:
                                    if dec_node.type == "decorator":
                                        decorators.append(
                                            dec_node.text.decode("utf-8").strip().lstrip("@")
                                        )
                                method.decorators = decorators
                                methods.append(method)

        return ClassInfo(
            name=name,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_code=source_code,
            docstring=docstring,
            base_classes=bases,
            methods=methods,
        )

    def _extract_python_import(
        self, node, source_bytes: bytes, file_path: str
    ) -> Optional[ImportInfo]:
        """Extract a Python import statement."""
        text = node.text.decode("utf-8")

        if node.type == "import_statement":
            # import foo, import foo.bar
            module_node = node.child_by_field_name("name")
            if module_node:
                return ImportInfo(
                    module=module_node.text.decode("utf-8"),
                    names=[],
                    file_path=file_path,
                    line_number=node.start_point[0] + 1,
                )
        elif node.type == "import_from_statement":
            # from foo import bar, baz
            module_node = node.child_by_field_name("module_name")
            module = module_node.text.decode("utf-8") if module_node else ""
            names = []
            for child in node.children:
                if child.type == "dotted_name" and child != module_node:
                    names.append(child.text.decode("utf-8"))
                elif child.type == "aliased_import":
                    name_part = child.child_by_field_name("name")
                    if name_part:
                        names.append(name_part.text.decode("utf-8"))
                elif child.type == "wildcard_import":
                    names.append("*")
            return ImportInfo(
                module=module,
                names=names,
                file_path=file_path,
                line_number=node.start_point[0] + 1,
            )

        return None

    def _extract_python_variable(
        self, node, source_bytes: bytes, file_path: str
    ) -> Optional[VariableInfo]:
        """Extract a module-level variable assignment."""
        # Look for assignment within expression_statement
        for child in node.children:
            if child.type == "assignment":
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                if left and left.type == "identifier":
                    name = left.text.decode("utf-8")
                    # Skip private/dunder
                    if name.startswith("_") and not name.startswith("__"):
                        return None
                    value = right.text.decode("utf-8")[:100] if right else None
                    return VariableInfo(
                        name=name,
                        file_path=file_path,
                        line_number=node.start_point[0] + 1,
                        value_preview=value,
                    )
        return None

    # ── JavaScript/TypeScript Extraction ─────────────────────────────────

    def _extract_javascript(self, root, source: str, parsed: ParsedFile):
        """Extract JavaScript/TypeScript constructs from AST."""
        source_bytes = bytes(source, "utf-8")
        self._walk_js_node(root, source_bytes, parsed, parent_class=None)

    def _walk_js_node(self, node, source_bytes: bytes, parsed: ParsedFile,
                      parent_class: Optional[str]):
        """Recursively walk JS/TS AST nodes."""
        for child in node.children:
            if child.type in ("function_declaration", "function"):
                func = self._extract_js_function(child, source_bytes, parsed.file_path)
                if func:
                    func.parent_class = parent_class
                    parsed.functions.append(func)

            elif child.type == "class_declaration":
                cls_name_node = child.child_by_field_name("name")
                cls_name = cls_name_node.text.decode("utf-8") if cls_name_node else "anonymous"
                cls = ClassInfo(
                    name=cls_name,
                    file_path=parsed.file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    source_code=child.text.decode("utf-8"),
                )
                parsed.classes.append(cls)
                # Recurse into class body for methods
                body = child.child_by_field_name("body")
                if body:
                    self._walk_js_node(body, source_bytes, parsed, parent_class=cls_name)

            elif child.type in ("lexical_declaration", "variable_declaration"):
                # Could contain arrow functions or variable assignments
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        value_node = decl.child_by_field_name("value")
                        if name_node and value_node:
                            if value_node.type == "arrow_function":
                                func = FunctionInfo(
                                    name=name_node.text.decode("utf-8"),
                                    file_path=parsed.file_path,
                                    line_start=child.start_point[0] + 1,
                                    line_end=child.end_point[0] + 1,
                                    source_code=child.text.decode("utf-8"),
                                    calls=self._extract_calls(value_node, source_bytes),
                                )
                                parsed.functions.append(func)

            elif child.type in ("import_statement", "import_declaration"):
                text = child.text.decode("utf-8")
                parsed.imports.append(ImportInfo(
                    module=text,
                    names=[],
                    file_path=parsed.file_path,
                    line_number=child.start_point[0] + 1,
                ))
            else:
                # Recurse
                if child.child_count > 0:
                    self._walk_js_node(child, source_bytes, parsed, parent_class=parent_class)

    def _extract_js_function(
        self, node, source_bytes: bytes, file_path: str
    ) -> Optional[FunctionInfo]:
        """Extract a JS/TS function."""
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "anonymous"

        return FunctionInfo(
            name=name,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_code=node.text.decode("utf-8"),
            calls=self._extract_calls(node, source_bytes),
        )

    # ── Generic Extraction ───────────────────────────────────────────────

    def _extract_generic(self, root, source: str, parsed: ParsedFile):
        """Fallback extraction for unsupported languages."""
        source_bytes = bytes(source, "utf-8")
        self._walk_generic(root, source_bytes, parsed)

    def _walk_generic(self, node, source_bytes: bytes, parsed: ParsedFile):
        """Walk tree and extract function/class definitions by node type name."""
        for child in node.children:
            if "function" in child.type and "definition" in child.type:
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8") if name_node else "unknown"
                parsed.functions.append(FunctionInfo(
                    name=name,
                    file_path=parsed.file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    source_code=child.text.decode("utf-8"),
                ))
            elif "class" in child.type and "definition" in child.type:
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8") if name_node else "unknown"
                parsed.classes.append(ClassInfo(
                    name=name,
                    file_path=parsed.file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    source_code=child.text.decode("utf-8"),
                ))
            if child.child_count > 0:
                self._walk_generic(child, source_bytes, parsed)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_python_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Extract docstring from a function or class body."""
        body = node.child_by_field_name("body")
        if not body or body.child_count == 0:
            return None

        first = body.children[0]
        if first.type == "expression_statement":
            expr = first.children[0] if first.child_count > 0 else None
            if expr and expr.type == "string":
                raw = expr.text.decode("utf-8")
                # Strip triple quotes
                for q in ('"""', "'''"):
                    if raw.startswith(q) and raw.endswith(q):
                        return raw[3:-3].strip()
                return raw.strip("\"'").strip()
        return None

    def _extract_calls(self, node, source_bytes: bytes) -> List[str]:
        """Recursively find all call_expression nodes and return callee names."""
        calls = []
        self._walk_calls(node, calls)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for c in calls:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result

    def _walk_calls(self, node, calls: List[str]):
        """Walk AST to find call expressions."""
        if node.type == "call":
            # Python: call node with function child
            func_node = node.child_by_field_name("function")
            if func_node:
                name = func_node.text.decode("utf-8")
                # Simplify attribute calls: obj.method -> method
                if "." in name:
                    name = name.rsplit(".", 1)[-1]
                calls.append(name)
        elif node.type == "call_expression":
            # JS/TS
            func_node = node.child_by_field_name("function")
            if func_node:
                name = func_node.text.decode("utf-8")
                if "." in name:
                    name = name.rsplit(".", 1)[-1]
                calls.append(name)

        for child in node.children:
            self._walk_calls(child, calls)
