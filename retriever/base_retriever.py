"""
Abstract base retriever.

All three retrieval systems (Vector, Vectorless, Hybrid) implement this
interface so experiments can swap strategies without changing any other code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from indexer.bm25_index import ScoredPage


@dataclass
class RetrievalResult:
    """
    Complete result from a retrieval operation.

    Contains ranked pages, metadata about which sources contributed,
    and timing information for efficiency evaluation.
    """
    query: str
    pages: List[ScoredPage]                     # Ranked results
    metadata: Dict = field(default_factory=dict) # Retriever-specific metadata
    retrieval_time_ms: float = 0.0               # Wall-clock time for retrieval
    strategy: str = ""                            # "vector" | "vectorless" | "hybrid"

    @property
    def page_ids(self) -> List[str]:
        return [sp.page.page_id for sp in self.pages]

    @property
    def file_paths(self) -> List[str]:
        seen = set()
        result = []
        for sp in self.pages:
            fp = sp.page.file_path
            if fp not in seen:
                seen.add(fp)
                result.append(fp)
        return result


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    Subclasses must implement `retrieve()`. The shared interface ensures
    that evaluation code works identically across all three systems.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve the top-k most relevant code pages for a query.

        Args:
            query: Natural language query or code identifier.
            top_k: Number of results to return.

        Returns:
            RetrievalResult with ranked pages and metadata.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this retriever."""
        ...
