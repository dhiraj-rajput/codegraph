"""
Query Rewriter — Uses local Ollama to expand natural language queries into
keyword-rich, structurally contextualized search tokens.

Ollama-only. No OpenAI, no Gemini, no HuggingFace.
"""

import json
import logging
from typing import List
from dataclasses import dataclass

from config.settings import DEFAULT_CONFIG, OLLAMA_HOST

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """The structured output of a rewritten query."""
    original_query: str
    keywords: List[str]
    expected_symbols: List[str]
    expected_files: List[str]
    hyde_snippet: str = ""


# ─── Expansion Prompt ────────────────────────────────────────────────────────

_EXPANSION_PROMPT = """You are a senior software engineer optimizing a search query for a Code Repository Retrieval System.
The user is asking: {query}

Rewrite this into structured arrays to maximize search recall.
Think about vocabulary mismatch: if they ask about "routing", the code might use `Router`, `endpoint`, `@app.get`, etc.

Output MUST be valid JSON:
{{
  "keywords": ["list", "of", "BM25", "search", "terms"],
  "expected_symbols": ["CamelCase", "or", "snake_case", "identifiers"],
  "expected_files": ["probable/file/paths.py"]
}}"""


class LLMQueryExpander:
    """
    Expands vague NL queries into structured search parameters using local Ollama.
    Falls back to regex extraction if Ollama is unavailable.
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.llm
        self._client = None

    def _init_client(self):
        """Lazy-init the Ollama client."""
        if self._client is not None:
            return
        try:
            from ollama import Client
            self._client = Client(host=OLLAMA_HOST)
        except (ImportError, Exception) as e:
            logger.warning(f"Ollama not available for query expansion: {e}")

    def expand(self, query: str) -> ExpandedQuery:
        """
        Rewrite the query using Ollama. Fallback to basic extraction if unavailable.
        """
        self._init_client()
        if self._client is None:
            return self._fallback(query)

        prompt = _EXPANSION_PROMPT.format(query=query)

        try:
            response = self._client.chat(
                model=self.config.fast_model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_predict": 512},
            )

            raw = response.get("message", {}).get("content", "{}")
            data = json.loads(raw)

            # Merge original query tokens into keywords for safety
            base_kws = query.lower().split()
            keywords = list(set(
                [k.lower() for k in data.get("keywords", [])] + base_kws
            ))

            eq = ExpandedQuery(
                original_query=query,
                keywords=keywords,
                expected_symbols=data.get("expected_symbols", []),
                expected_files=data.get("expected_files", []),
            )
            logger.info(f"Expanded query: symbols={len(eq.expected_symbols)} kws={len(eq.keywords)}")
            return eq

        except Exception as e:
            logger.error(f"LLM query expansion failed: {e}")
            return self._fallback(query)

    @staticmethod
    def _fallback(query: str) -> ExpandedQuery:
        """Regex-based fallback when Ollama is not available."""
        from retriever.vectorless_retriever import extract_identifiers, extract_keywords
        return ExpandedQuery(
            original_query=query,
            keywords=extract_keywords(query),
            expected_symbols=extract_identifiers(query),
            expected_files=[],
        )
