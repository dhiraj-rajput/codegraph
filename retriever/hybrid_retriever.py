"""
System C: Hybrid Retriever.

Combines sparse (BM25), dense (vector), and structural (graph) signals
using Reciprocal Rank Fusion (RRF) for result merging.

Pipeline:
  query → BM25 search + vector search + symbol lookup
        → Reciprocal Rank Fusion
        → graph expansion
        → re-rank
        → top-k pages

Optimizations:
  - Cached PageRank (computed once at build, not per query)
  - Batched shortest-path for graph proximity scoring
  - Dead methods _symbol_boost and _graph_expand removed
"""

import time
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from retriever.base_retriever import BaseRetriever, RetrievalResult
from retriever.vectorless_retriever import extract_identifiers, extract_keywords
from indexer.bm25_index import BM25CodeIndex, ScoredPage
from indexer.symbol_index import SymbolIndex
from indexer.page_index import PageIndex
from indexer.vector_index import VectorCodeIndex
from graph_builder.code_graph import CodeGraph
from config.settings import DEFAULT_CONFIG
from retriever.query_rewriter import LLMQueryExpander

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[ScoredPage]],
    k: int = 60,
) -> List[ScoredPage]:
    """
    Reciprocal Rank Fusion (RRF) for merging multiple ranked result lists.

    RRF score = Σ 1 / (k + rank_i) for each list i

    This is the standard fusion method used in hybrid retrieval research
    (Cormack et al., 2009).
    """
    page_scores: Dict[str, float] = defaultdict(float)
    page_objects: Dict[str, ScoredPage] = {}

    for ranked_list in ranked_lists:
        for rank, sp in enumerate(ranked_list):
            pid = sp.page.page_id
            page_scores[pid] += 1.0 / (k + rank + 1)
            if pid not in page_objects:
                page_objects[pid] = sp

    # Sort by fused score
    sorted_pids = sorted(page_scores.keys(), key=lambda x: page_scores[x], reverse=True)

    result = []
    for pid in sorted_pids:
        sp = page_objects[pid]
        result.append(ScoredPage(
            page=sp.page,
            score=page_scores[pid],
            source="hybrid_rrf",
        ))

    return result


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval (System C).

    Combines all three signal types:
    1. BM25 lexical search (sparse)
    2. Vector similarity search (dense)
    3. Graph expansion (structural)

    Uses Reciprocal Rank Fusion (RRF) to merge BM25 and vector results,
    then expands with graph traversal.
    """

    def __init__(
        self,
        bm25_index: BM25CodeIndex,
        vector_index: VectorCodeIndex,
        symbol_index: SymbolIndex,
        page_index: PageIndex,
        code_graph: CodeGraph,
        config=None,
    ):
        self._bm25 = bm25_index
        self._vector = vector_index
        self._symbols = symbol_index
        self._pages = page_index
        self._graph = code_graph
        cfg = config or DEFAULT_CONFIG.retriever
        llm_cfg = DEFAULT_CONFIG.llm
        self._bm25_weight = cfg.hybrid_bm25_weight
        self._vector_weight = cfg.hybrid_vector_weight
        self._graph_weight = cfg.hybrid_graph_weight
        self._graph_depth = cfg.graph_depth

        self._expander = None
        if llm_cfg.enable_query_expansion:
            self._expander = LLMQueryExpander(config=llm_cfg)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()

        # Step 1: Expand NLP query
        if self._expander:
            expanded = self._expander.expand(query)
            identifiers = expanded.expected_symbols
            search_query = " ".join(expanded.keywords) + " " + " ".join(identifiers)
        else:
            identifiers = extract_identifiers(query)
            search_query = query

        # Step 2: Vector Search (Semantic Seed Gathering)
        vector_results = self._vector.search(query, top_k=top_k)

        # Step 3: Graph Expansion from Vector Seeds
        seed_symbol_ids = []
        for sp in vector_results:
            if sp.page.symbol_id:
                seed_symbol_ids.append(sp.page.symbol_id)

        # Add exact symbol lookup hits as supplementary graph seeds
        symbol_hits = []
        for ident in identifiers:
            symbol_hits.extend(self._symbols.lookup(ident))
        for doc in symbol_hits:
            seed_symbol_ids.append(doc["symbol_id"])

        seed_symbol_ids = list(set(seed_symbol_ids))[:10]

        # Expand Graph — OPTIMIZED with cached PageRank + batched BFS
        neighbors = self._graph.expand_graph(
            seed_symbol_ids, depth=self._graph_depth, max_nodes=30
        )

        # Cached PageRank — O(1), not O(V+E) per query
        graph_pageranks = self._graph.get_pagerank()

        graph_pages = []
        for sym in neighbors:
            page = self._pages.get_by_symbol(sym.symbol_id)
            if page:
                pr_score = graph_pageranks.get(sym.symbol_id, 0.0) * 100
                graph_pages.append(ScoredPage(page=page, score=pr_score, source="graph"))

        # Step 4: Run BM25 search for sparse lexical completeness
        bm25_results = self._bm25.search(search_query, top_k=top_k)

        # Step 5: Merge all pipelines using RRF
        fused = reciprocal_rank_fusion([bm25_results, vector_results, graph_pages])

        # Step 6: Final re-rank
        fused.sort(key=lambda x: x.score, reverse=True)
        fused = fused[:top_k]

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            pages=fused,
            metadata={
                "source": "hybrid",
                "bm25_hits": len(bm25_results),
                "vector_hits": len(vector_results),
                "identifiers": identifiers,
                "symbol_boosts": len(symbol_hits),
            },
            retrieval_time_ms=elapsed,
            strategy="hybrid",
        )

    @property
    def name(self) -> str:
        return "Hybrid RAG (System C)"
