"""
System A: Vector Retriever.

Pure vector similarity search baseline. Embeds the query and retrieves
the most similar code pages from the vector index.
"""

import time
import logging
from typing import Optional

from retriever.base_retriever import BaseRetriever, RetrievalResult
from indexer.vector_index import VectorCodeIndex

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """
    Vector-only retrieval (System A baseline).

    Pipeline: query → embed → cosine similarity → top-k pages
    """

    def __init__(self, vector_index: VectorCodeIndex):
        self._vector_index = vector_index

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()

        pages = self._vector_index.search(query, top_k=top_k)

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            pages=pages,
            metadata={
                "source": "vector",
                "vector_results": len(pages),
            },
            retrieval_time_ms=elapsed,
            strategy="vector",
        )

    @property
    def name(self) -> str:
        return "Vector RAG (System A)"
