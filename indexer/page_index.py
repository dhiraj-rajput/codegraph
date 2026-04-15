"""
Function-level page index.

Each code entity (function, class, file) becomes a "page" — the atomic
retrieval unit. This replaces arbitrary 512-token chunking with
semantically meaningful code boundaries.
"""

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict

from parser.symbol_extractor import CodeSymbol
from parser.tree_sitter_parser import ParsedFile


@dataclass
class CodePage:
    """
    A single retrievable unit of code.

    Each page corresponds to a function, class, or file — never an
    arbitrary token-window chunk.
    """
    page_id: str
    symbol_id: str                   # Links back to CodeSymbol
    symbol_name: str
    qualified_name: str
    symbol_type: str                 # function | class | method | file
    file_path: str
    source_code: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    token_count: int = 0
    language: str = "python"
    calls: List[str] = field(default_factory=list)

    @property
    def searchable_text(self) -> str:
        """Combined text for BM25/vector indexing."""
        parts = [self.symbol_name, self.qualified_name]
        if self.docstring:
            parts.append(self.docstring)
        if self.signature:
            parts.append(self.signature)
        parts.append(self.source_code)
        parts.append(self.file_path)
        return "\n".join(parts)


class PageIndex:
    """
    Manages the creation and lookup of CodePage objects.

    Converts CodeSymbol records into pages with token counting,
    and provides fast lookup by page_id, symbol_id, file, or name.
    """

    def __init__(self):
        self._pages: Dict[str, CodePage] = {}           # page_id -> CodePage
        self._by_symbol: Dict[str, str] = {}             # symbol_id -> page_id
        self._by_file: Dict[str, List[str]] = {}         # file_path -> [page_ids]
        self._by_name: Dict[str, List[str]] = {}         # name -> [page_ids]

    def build(self, symbols: List[CodeSymbol], max_tokens: int = 512):
        """
        Build page index from symbols.

        Each function/method = 1 page.
        Each class = 1 page (full source).
        Duplicate methods already covered by class page are kept
        as separate pages for fine-grained retrieval.
        """
        self._pages.clear()
        self._by_symbol.clear()
        self._by_file.clear()
        self._by_name.clear()

        for sym in symbols:
            if sym.type in ("function", "method", "class"):
                page = self._symbol_to_page(sym)
                self._add_page(page)
            elif sym.type == "variable" and len(sym.source_code) > 20:
                # Include significant variable assignments
                page = self._symbol_to_page(sym)
                self._add_page(page)

    def build_from_files(self, parsed_files: List[ParsedFile]):
        """
        Build file-level pages for files that have no functions/classes.

        Call this after build() to catch "loose" files.
        """
        files_with_symbols = set()
        for page in self._pages.values():
            files_with_symbols.add(page.file_path)

        for pf in parsed_files:
            if pf.file_path not in files_with_symbols:
                # Create a file-level page
                page = CodePage(
                    page_id=hashlib.sha256(
                        f"file::{pf.file_path}".encode()
                    ).hexdigest()[:16],
                    symbol_id=f"file::{pf.file_path}",
                    symbol_name=pf.file_path.split("/")[-1].split("\\")[-1],
                    qualified_name=pf.file_path,
                    symbol_type="file",
                    file_path=pf.file_path,
                    source_code=pf.source_code[:2000],  # Truncate large files
                    line_start=1,
                    line_end=pf.line_count,
                    token_count=len(pf.source_code) // 4,
                    language=pf.language,
                )
                self._add_page(page)

    def _symbol_to_page(self, sym: CodeSymbol) -> CodePage:
        """Convert a CodeSymbol to a CodePage."""
        return CodePage(
            page_id=sym.symbol_id,
            symbol_id=sym.symbol_id,
            symbol_name=sym.name,
            qualified_name=sym.qualified_name,
            symbol_type=sym.type,
            file_path=sym.file_path,
            source_code=sym.source_code,
            docstring=sym.docstring,
            signature=sym.signature,
            line_start=sym.line_start,
            line_end=sym.line_end,
            token_count=sym.token_estimate,
            language=sym.language,
            calls=sym.calls,
        )

    def _add_page(self, page: CodePage):
        """Add a page to all indexes."""
        self._pages[page.page_id] = page
        self._by_symbol[page.symbol_id] = page.page_id

        if page.file_path not in self._by_file:
            self._by_file[page.file_path] = []
        self._by_file[page.file_path].append(page.page_id)

        if page.symbol_name not in self._by_name:
            self._by_name[page.symbol_name] = []
        self._by_name[page.symbol_name].append(page.page_id)

    # ── Lookup ───────────────────────────────────────────────────────────

    def get_page(self, page_id: str) -> Optional[CodePage]:
        return self._pages.get(page_id)

    def get_by_symbol(self, symbol_id: str) -> Optional[CodePage]:
        pid = self._by_symbol.get(symbol_id)
        return self._pages.get(pid) if pid else None

    def get_by_file(self, file_path: str) -> List[CodePage]:
        pids = self._by_file.get(file_path, [])
        return [self._pages[pid] for pid in pids if pid in self._pages]

    def get_by_name(self, name: str) -> List[CodePage]:
        pids = self._by_name.get(name, [])
        return [self._pages[pid] for pid in pids if pid in self._pages]

    @property
    def all_pages(self) -> List[CodePage]:
        return list(self._pages.values())

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def save(self, path: str):
        """Save PageIndex to disk as JSON (safer than pickle)."""
        data = {pid: asdict(page) for pid, page in self._pages.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load PageIndex from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._pages = {pid: CodePage(**page) for pid, page in raw.items()}
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Backward compatibility for old pickle indexes.
            with open(path, "rb") as f:
                self._pages = pickle.load(f)
        
        # Rebuild lookup maps
        self._by_symbol.clear()
        self._by_file.clear()
        self._by_name.clear()
        for pid, page in self._pages.items():
            self._by_symbol[page.symbol_id] = pid
            if page.file_path not in self._by_file:
                self._by_file[page.file_path] = []
            self._by_file[page.file_path].append(pid)
            if page.symbol_name not in self._by_name:
                self._by_name[page.symbol_name] = []
            self._by_name[page.symbol_name].append(pid)
