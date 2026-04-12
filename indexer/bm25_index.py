"""
BM25 index for code pages.

Uses rank-bm25 with code-aware tokenization (splits camelCase, snake_case,
preserves identifiers). This is the primary sparse retrieval algorithm
for the vectorless RAG pipeline.
"""

import re
import logging
import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from indexer.page_index import CodePage

logger = logging.getLogger(__name__)


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
    words = re.split(r'[\s\(\)\{\}\[\]:,;=\+\-\*/\<\>\!\&\|\^\~\%\#\"\'\`]+', text)

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

    Documents are function-level code pages with code-aware tokenization.
    Supports both BM25Okapi (standard) retrieval.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._pages: List[CodePage] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, pages: List[CodePage]):
        """
        Build the BM25 index from a list of code pages.

        Each page's searchable_text is tokenized with code-aware splitting.
        """
        self._pages = pages
        self._tokenized_corpus = [
            tokenize_code(page.searchable_text) for page in pages
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Built BM25 index with {len(pages)} pages")

    def search(self, query: str, top_k: int = 10) -> List[ScoredPage]:
        """
        Search the BM25 index with a query string.

        Returns top-k ScoredPage results ordered by BM25 score (descending).
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built yet")
            return []

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

        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        return [(self._pages[i].page_id, float(s)) for i, s in enumerate(scores)]

    def save(self, path: str):
        """Save BM25 index to disk."""
        data = {
            "tokenized_corpus": self._tokenized_corpus,
            "pages": self._pages,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._pages = data["pages"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @property
    def page_count(self) -> int:
        return len(self._pages)
