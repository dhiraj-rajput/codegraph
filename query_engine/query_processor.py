"""
Query processor — transforms natural language queries into structured
search requests with extracted keywords, identifiers, and query classification.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class QueryType(str, Enum):
    """Query classification for routing and prompt selection."""
    ARCHITECTURE = "architecture"    # "How does X work?"
    DEPENDENCY = "dependency"        # "What calls X?" / "Where is X used?"
    DEBUGGING = "debugging"          # "Why might X fail?"
    GENERAL = "general"


@dataclass
class ProcessedQuery:
    """A processed, structured query ready for retrieval."""
    raw_query: str
    keywords: List[str]
    identifiers: List[str]
    query_type: QueryType
    expanded_terms: List[str] = field(default_factory=list)


# ─── Patterns ────────────────────────────────────────────────────────────────

_CAMEL = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+\b')
_LOWER_CAMEL = re.compile(r'\b[a-z]+(?:[A-Z][a-zA-Z]*)+\b')
_SNAKE = re.compile(r'\b[a-z]+(?:_[a-z0-9]+)+\b')
_UPPER_SNAKE = re.compile(r'\b[A-Z]+(?:_[A-Z0-9]+)+\b')
_DOTTED = re.compile(r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b')
_BACKTICK = re.compile(r'`([^`]+)`')

_ARCH_WORDS = {"how", "work", "works", "architecture", "design", "structure",
               "implement", "implemented", "flow", "lifecycle", "explain"}
_DEP_WORDS = {"call", "calls", "called", "import", "imports", "imported",
              "use", "uses", "used", "depend", "depends", "dependency",
              "where", "which", "what"}
_DEBUG_WORDS = {"why", "fail", "fails", "error", "bug", "crash", "wrong",
                "broken", "issue", "problem", "cause", "exception", "fix"}

_STOP_WORDS = {
    "how", "does", "what", "where", "when", "why", "which", "who",
    "is", "are", "was", "were", "the", "a", "an", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "this", "that", "it",
    "can", "could", "would", "should", "do", "did", "has", "have",
    "been", "be", "will", "shall", "may", "might", "not", "and", "or",
    "but", "if", "then", "so", "about", "into", "through",
}


class QueryProcessor:
    """
    Processes natural language queries into structured search inputs.

    Steps:
    1. Extract code identifiers (camelCase, snake_case, backtick-quoted)
    2. Extract meaningful keywords
    3. Classify query type (architecture/dependency/debugging)
    4. Optionally expand query with related terms
    """

    def process(self, raw_query: str) -> ProcessedQuery:
        """Process a raw query string into a structured query."""
        identifiers = self._extract_identifiers(raw_query)
        keywords = self._extract_keywords(raw_query)
        query_type = self._classify_query(raw_query)

        return ProcessedQuery(
            raw_query=raw_query,
            keywords=keywords,
            identifiers=identifiers,
            query_type=query_type,
        )

    def _extract_identifiers(self, text: str) -> List[str]:
        """Extract code identifiers from text."""
        idents = set()

        # Backtick-quoted identifiers (highest priority)
        for match in _BACKTICK.finditer(text):
            idents.add(match.group(1))

        # Regex-based detection
        for pattern in [_CAMEL, _LOWER_CAMEL, _SNAKE, _UPPER_SNAKE, _DOTTED]:
            for match in pattern.finditer(text):
                ident = match.group()
                idents.add(ident)
                if "." in ident:
                    idents.add(ident.rsplit(".", 1)[-1])

        return list(idents)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords, excluding stopwords."""
        words = re.split(r'[\s\?\.\,\!\;\:\`]+', text.lower())
        return [w for w in words if w and len(w) > 2 and w not in _STOP_WORDS]

    def _classify_query(self, text: str) -> QueryType:
        """Classify query into architecture, dependency, or debugging."""
        lower = text.lower()
        words = set(lower.split())

        arch_score = len(words & _ARCH_WORDS)
        dep_score = len(words & _DEP_WORDS)
        debug_score = len(words & _DEBUG_WORDS)

        if debug_score > arch_score and debug_score > dep_score:
            return QueryType.DEBUGGING
        if dep_score > arch_score:
            return QueryType.DEPENDENCY
        if arch_score > 0:
            return QueryType.ARCHITECTURE
        return QueryType.GENERAL
