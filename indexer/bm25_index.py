"""
BM25 index for code pages.

Uses bm25s (sparse matrix BM25) for up to 500x faster scoring than rank-bm25.
Falls back to rank-bm25 if bm25s is not installed.

Code-aware tokenization: splits camelCase, snake_case, preserves identifiers.
"""

import re
import logging
import json
import pickle
from typing import List, Tuple, Optional
from dataclasses import asdict, dataclass

from indexer.page_index import CodePage

logger = logging.getLogger(__name__)

# Try bm25s first (500x faster via sparse matrices), fall back to rank-bm25
try:
    import bm25s
    BM25S_AVAILABLE = True
    logger.debug("Using bm25s (fast sparse BM25)")
except ImportError:
    BM25S_AVAILABLE = False
    from rank_bm25 import BM25Okapi
    logger.debug("bm25s not installed, using rank-bm25 (slower)")


@dataclass
class ScoredPage:
    """A code page with a retrieval score."""
    page: CodePage
    score: float
    source: str = "bm25"            # Which retriever produced this result


# ─── Code-Aware Tokenizer ───────────────────────────────────────────────────

# Common noise tokens in code that hurt BM25 precision
CODE_STOPWORDS = {
    "self", "cls", "return", "import", "from", "def", "class", "if", "else",
    "elif", "for", "while", "try", "except", "finally", "with", "as", "in",
    "is", "not", "and", "or", "none", "true", "false", "pass", "break",
    "continue", "raise", "yield", "lambda", "global", "nonlocal", "assert",
    "del", "print", "the", "a", "an", "of", "to", "this", "that",
    # Brackets and operators as single tokens after splitting
    "(", ")", "{", "}", "[", "]", ":", ",", ".", "=", "==", "!=",
    "->", "**", "//", "+=", "-=", "*=", "/=",
}

_CAMEL_PATTERN = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
_SNAKE_PATTERN = re.compile(r'_+')
_NONALPHA = re.compile(r'[^a-zA-Z0-9_]')
_SPLIT_PATTERN = re.compile(r'[\s\(\)\{\}\[\]:,;=\+\-\*/\<\>\!\&\|\^\~\%\#\"\'\`]+')


def tokenize_code(text: str) -> List[str]:
    """
    Code-aware tokenizer for BM25.

    1. Split on whitespace and non-alphanumeric characters.
    2. Split camelCase into subwords (validateJWT -> validate, jwt).
    3. Split snake_case into subwords (auth_middleware -> auth, middleware).
    4. Keep original identifiers as tokens too.
    5. Remove code stopwords.
    6. Lowercase everything.
    """
    tokens = []

    # Basic split on whitespace and punctuation
    words = _SPLIT_PATTERN.split(text)

    for word in words:
        if not word:
            continue

        lower = word.lower()

        # Skip stopwords
        if lower in CODE_STOPWORDS:
            continue

        # Add the full identifier
        if len(lower) > 1:
            tokens.append(lower)

        # Split camelCase
        camel_parts = _CAMEL_PATTERN.split(word)
        if len(camel_parts) > 1:
            for part in camel_parts:
                p = part.lower()
                if len(p) > 1 and p not in CODE_STOPWORDS:
                    tokens.append(p)

        # Split snake_case (if contains underscore)
        if "_" in word:
            snake_parts = _SNAKE_PATTERN.split(word)
            for part in snake_parts:
                p = part.lower()
                if len(p) > 1 and p not in CODE_STOPWORDS:
                    tokens.append(p)

    return tokens


# ─── BM25 Index ─────────────────────────────────────────────────────────────

class BM25CodeIndex:
    """
    BM25 index over code pages.

    Supports two backends:
    - bm25s (preferred): Uses SciPy sparse matrices for 500x faster scoring.
      Pre-computes all BM25 term-document scores at index time.
    - rank-bm25 (fallback): Standard BM25Okapi if bm25s is not installed.
    """

    def __init__(self):
        self._bm25 = None
        self._pages: List[CodePage] = []
        self._tokenized_corpus: List[List[str]] = []
        self._use_bm25s = BM25S_AVAILABLE

    def build(self, pages: List[CodePage]):
        """
        Build the BM25 index from a list of code pages.

        Each page's searchable_text is tokenized with code-aware splitting.
        """
        self._pages = pages
        self._tokenized_corpus = [
            tokenize_code(page.searchable_text) for page in pages
        ]

        if self._use_bm25s:
            # bm25s: pre-compute sparse score matrix at index time
            self._bm25 = bm25s.BM25()
            # bm25s expects a corpus as list of lists of strings
            self._bm25.index(bm25s.tokenize(
                [page.searchable_text for page in pages],
                stopwords=list(CODE_STOPWORDS),
            ))
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"Built BM25 index with {len(pages)} pages"
                     f" (backend={'bm25s' if self._use_bm25s else 'rank-bm25'})")

    def search(self, query: str, top_k: int = 10) -> List[ScoredPage]:
        """
        Search the BM25 index with a query string.

        Returns top-k ScoredPage results ordered by BM25 score (descending).
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built yet")
            return []

        if self._use_bm25s:
            query_tokens = bm25s.tokenize(query, stopwords=list(CODE_STOPWORDS))
            results, scores = self._bm25.retrieve(query_tokens, k=min(top_k, len(self._pages)))

            scored = []
            for i in range(results.shape[1]):
                idx = int(results[0, i])
                score = float(scores[0, i])
                if score > 0 and idx < len(self._pages):
                    scored.append(ScoredPage(
                        page=self._pages[idx],
                        score=score,
                        source="bm25",
                    ))
            return scored
        else:
            # rank-bm25 fallback
            query_tokens = tokenize_code(query)
            if not query_tokens:
                return []

            scores = self._bm25.get_scores(query_tokens)

            # Get top-k indices
            scored_indices = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:top_k]

            results = []
            for idx, score in scored_indices:
                if score > 0:
                    results.append(ScoredPage(
                        page=self._pages[idx],
                        score=float(score),
                        source="bm25",
                    ))

            return results

    def get_scores(self, query: str) -> List[Tuple[str, float]]:
        """Get BM25 scores for all pages (for hybrid fusion)."""
        if self._bm25 is None:
            return []

        if self._use_bm25s:
            query_tokens = bm25s.tokenize(query, stopwords=list(CODE_STOPWORDS))
            results, scores = self._bm25.retrieve(query_tokens, k=min(100, len(self._pages)))
            output = []
            for i in range(results.shape[1]):
                idx = int(results[0, i])
                score = float(scores[0, i])
                if idx < len(self._pages):
                    output.append((self._pages[idx].page_id, score))
            return output
        else:
            query_tokens = tokenize_code(query)
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(query_tokens)
            return [(self._pages[i].page_id, float(s)) for i, s in enumerate(scores)]

    def save(self, path: str):
        """Save BM25 index metadata to disk as JSON."""
        data = {
            "tokenized_corpus": self._tokenized_corpus,
            "pages": [asdict(p) for p in self._pages],
            "use_bm25s": self._use_bm25s,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load BM25 index metadata from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._pages = [CodePage(**p) for p in data["pages"]]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._use_bm25s = data.get("use_bm25s", False) and BM25S_AVAILABLE
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Backward compatibility for old pickle indexes.
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._pages = data["pages"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._use_bm25s = data.get("use_bm25s", False) and BM25S_AVAILABLE

        if self._use_bm25s:
            self._bm25 = bm25s.BM25()
            self._bm25.index(bm25s.tokenize(
                [page.searchable_text for page in self._pages],
                stopwords=list(CODE_STOPWORDS),
            ))
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    @property
    def page_count(self) -> int:
        return len(self._pages)
