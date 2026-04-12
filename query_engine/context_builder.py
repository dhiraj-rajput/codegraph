"""
Context builder — assembles LLM context from retrieved pages.

Respects a token budget and prioritizes pages by retrieval score.
Includes call chains, file paths, and structural metadata.
"""

import logging
from typing import List, Optional

from indexer.bm25_index import ScoredPage
from graph_builder.code_graph import CodeGraph
from config.settings import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token). Use tiktoken for precision."""
    return len(text) // 4


class ContextBuilder:
    """
    Builds an LLM-ready context string from retrieved code pages.

    Includes:
    - File path + line numbers
    - Function/class signature
    - Docstring
    - Full source code (budget permitting)
    - Call chain from graph
    """

    def __init__(self, code_graph: Optional[CodeGraph] = None, config=None):
        self._graph = code_graph
        cfg = config or DEFAULT_CONFIG.llm
        self._budget = cfg.token_budget

    def build(
        self,
        pages: List[ScoredPage],
        query: str = "",
        budget: Optional[int] = None,
    ) -> str:
        """
        Assemble context from ranked pages within token budget.

        Pages are added in score order until the budget is exhausted.
        """
        target_budget = budget or self._budget
        context_parts = []
        used_tokens = 0

        for i, sp in enumerate(pages):
            page = sp.page
            section = self._format_page(page, rank=i + 1)
            section_tokens = count_tokens(section)

            if used_tokens + section_tokens > target_budget:
                # Try a truncated version
                truncated = self._format_page_truncated(page, rank=i + 1)
                trunc_tokens = count_tokens(truncated)
                if used_tokens + trunc_tokens <= target_budget:
                    context_parts.append(truncated)
                    used_tokens += trunc_tokens
                break

            context_parts.append(section)
            used_tokens += section_tokens

        context = "\n\n".join(context_parts)

        logger.debug(f"Built context: {len(pages)} pages considered, "
                     f"{len(context_parts)} included, ~{used_tokens} tokens")

        return context

    def _format_page(self, page, rank: int) -> str:
        """Format a full page with all metadata."""
        parts = [f"### [{rank}] {page.file_path}:{page.line_start}-{page.line_end}"]

        if page.signature:
            parts.append(f"**Signature:** `{page.signature}`")

        if page.docstring:
            parts.append(f"**Docstring:** {page.docstring[:200]}")

        # Call chain from graph
        if self._graph and page.calls:
            calls_str = ", ".join(page.calls[:10])
            parts.append(f"**Calls:** {calls_str}")

        # Full source code
        lang = page.language or "python"
        parts.append(f"```{lang}\n{page.source_code}\n```")

        return "\n".join(parts)

    def _format_page_truncated(self, page, rank: int) -> str:
        """Format a truncated page (signature + docstring only)."""
        parts = [f"### [{rank}] {page.file_path}:{page.line_start}-{page.line_end}"]

        if page.signature:
            parts.append(f"**Signature:** `{page.signature}`")

        if page.docstring:
            parts.append(f"**Docstring:** {page.docstring[:150]}")

        if page.calls:
            parts.append(f"**Calls:** {', '.join(page.calls[:5])}")

        return "\n".join(parts)
