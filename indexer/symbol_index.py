"""
Symbol index using SQLite for fast exact/fuzzy symbol lookup.

Provides O(1) lookup by name and LIKE-based pattern matching for
partial identifier resolution during query processing.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Optional

from parser.symbol_extractor import CodeSymbol
from indexer.page_index import CodePage

logger = logging.getLogger(__name__)


class SymbolIndex:
    """
    SQLite-backed symbol lookup table.

    Indexes symbol names, qualified names, and types for fast retrieval.
    Used by the vectorless retriever to resolve code identifiers in queries.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create the symbol lookup table."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                signature TEXT,
                docstring TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_qualified ON symbols(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_type ON symbols(type);
            CREATE INDEX IF NOT EXISTS idx_file ON symbols(file_path);
        """)
        self._conn.commit()

    def build(self, symbols: List[CodeSymbol]):
        """
        Populate the symbol index from CodeSymbol records.

        Clears existing data and rebuilds from scratch.
        """
        self._conn.execute("DELETE FROM symbols")

        rows = [
            (
                sym.symbol_id,
                sym.name,
                sym.qualified_name,
                sym.type,
                sym.file_path,
                sym.line_start,
                sym.line_end,
                sym.signature,
                sym.docstring,
            )
            for sym in symbols
        ]

        self._conn.executemany(
            """INSERT OR REPLACE INTO symbols
               (symbol_id, name, qualified_name, type, file_path,
                line_start, line_end, signature, docstring)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        logger.info(f"Built symbol index with {len(rows)} entries")

    def lookup(self, name: str) -> List[dict]:
        """
        Exact match on symbol name or qualified name.

        Returns list of dicts with symbol metadata.
        """
        cursor = self._conn.execute(
            """SELECT * FROM symbols
               WHERE name = ? OR qualified_name = ?""",
            (name, name),
        )
        return [dict(row) for row in cursor.fetchall()]

    def lookup_fuzzy(self, name: str) -> List[dict]:
        """
        Fuzzy lookup using LIKE patterns.

        Matches symbols where name contains the query string.
        """
        pattern = f"%{name}%"
        cursor = self._conn.execute(
            """SELECT * FROM symbols
               WHERE name LIKE ? OR qualified_name LIKE ?
               ORDER BY length(name) ASC
               LIMIT 20""",
            (pattern, pattern),
        )
        return [dict(row) for row in cursor.fetchall()]

    def lookup_by_type(self, sym_type: str) -> List[dict]:
        """Get all symbols of a given type."""
        cursor = self._conn.execute(
            "SELECT * FROM symbols WHERE type = ?", (sym_type,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def lookup_in_file(self, file_path: str) -> List[dict]:
        """Get all symbols in a given file."""
        cursor = self._conn.execute(
            "SELECT * FROM symbols WHERE file_path = ?", (file_path,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_symbol_ids(self, name: str) -> List[str]:
        """Get symbol IDs matching a name (for graph lookup)."""
        cursor = self._conn.execute(
            """SELECT symbol_id FROM symbols
               WHERE name = ? OR qualified_name = ?""",
            (name, name),
        )
        return [row[0] for row in cursor.fetchall()]

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Search symbols matching any word in the query.

        Used by the retriever to find symbols referenced in user queries.
        """
        words = query.split()
        if not words:
            return []

        # Build OR query for each word
        conditions = []
        params = []
        for word in words:
            if len(word) > 2:
                conditions.append("(name LIKE ? OR qualified_name LIKE ?)")
                params.extend([f"%{word}%", f"%{word}%"])

        if not conditions:
            return []

        sql = f"""SELECT *, COUNT(*) as match_count FROM symbols
                  WHERE {' OR '.join(conditions)}
                  GROUP BY symbol_id
                  ORDER BY match_count DESC, length(name) ASC
                  LIMIT ?"""

        params = tuple(params) + (int(top_k),)
        cursor = self._conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    @property
    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM symbols")
        return cursor.fetchone()[0]

    def close(self):
        self._conn.close()

    def save(self, path: str):
        """Save to a persistent SQLite file."""
        if self.db_path == ":memory:":
            target = sqlite3.connect(path)
            self._conn.backup(target)
            target.close()
            logger.info(f"Saved symbol index to {path}")

    def load(self, path: str):
        """Load from a persistent SQLite file."""
        self._conn.close()
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        logger.info(f"Loaded symbol index from {path}")
