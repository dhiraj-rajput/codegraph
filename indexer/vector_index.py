"""
Vector embedding index with state-of-the-art HuggingFace models.

Uses community-rated best open-source models for code embedding:

  Tier 1 (Recommended):
    • nomic-ai/nomic-embed-code     — Code-specialized, trained on CoRNStack
    • BAAI/bge-m3                   — Hybrid-ready (dense + sparse), 8192 token context

  Tier 2 (Lightweight fallback):
    • all-MiniLM-L6-v2              — Fast, CPU-friendly, general purpose

Embeddings are generated locally via sentence-transformers (100% free, no API keys).
Storage uses ChromaDB with persistent disk-backed HNSW index.
"""

import logging
import time
from typing import List, Optional, Dict
from dataclasses import dataclass

from indexer.page_index import CodePage
from indexer.bm25_index import ScoredPage

logger = logging.getLogger(__name__)


# ─── Model Registry ─────────────────────────────────────────────────────────

EMBEDDING_MODELS = {
    # ── Tier 1: Best-in-class for code (community-rated, free, open-source) ──
    "nomic-code": {
        "hf_name": "nomic-ai/nomic-embed-code",
        "dimension": 768,
        "max_tokens": 8192,
        "trust_remote_code": True,
        "description": "SOTA open-source code embedding. Trained on CoRNStack code/docstring pairs.",
        "query_prefix": "",               # No prefix needed for code model
        "document_prefix": "",
    },
    "bge-m3": {
        "hf_name": "BAAI/bge-m3",
        "dimension": 1024,
        "max_tokens": 8192,
        "trust_remote_code": False,
        "description": "Multi-modal retrieval (dense + sparse + ColBERT). MIT license.",
        "query_prefix": "",
        "document_prefix": "",
    },

    # ── Tier 2: Good general-purpose models with strong code performance ──
    "nomic-text": {
        "hf_name": "nomic-ai/nomic-embed-text-v2-moe",
        "dimension": 768,
        "max_tokens": 8192,
        "trust_remote_code": True,
        "description": "MoE text embedding. Needs search_query:/search_document: prefixes.",
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
    },
    "bge-base": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "max_tokens": 512,
        "trust_remote_code": False,
        "description": "Solid all-rounder. Smaller footprint than bge-m3.",
        "query_prefix": "Represent this sentence: ",
        "document_prefix": "",
    },

    # ── Tier 3: Lightweight / CPU fallback ──
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_tokens": 256,
        "trust_remote_code": False,
        "description": "Tiny, fast, CPU-friendly. Use for prototyping only.",
        "query_prefix": "",
        "document_prefix": "",
    },
}

DEFAULT_MODEL = "nomic-code"   # Best free model for code retrieval


# ─── Embedding Engine ────────────────────────────────────────────────────────

class HuggingFaceEmbedder:
    """
    Wraps sentence-transformers to embed code with any HuggingFace model.

    Loads models lazily, caches them, and handles query/document prefixes.
    100% local, 100% free — no API keys needed.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL):
        self._model_key = model_key
        self._model_info = EMBEDDING_MODELS[model_key]
        self._model = None
        self._load_time_sec = 0.0

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer
        import torch

        hf_name = self._model_info["hf_name"]
        trust = self._model_info.get("trust_remote_code", False)

        # Auto-detect best device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Loading embedding model: {hf_name} on {device} ...")
        start = time.perf_counter()

        self._model = SentenceTransformer(
            hf_name,
            device=device,
            trust_remote_code=trust,
        )

        self._load_time_sec = time.perf_counter() - start
        logger.info(
            f"Loaded {hf_name} in {self._load_time_sec:.1f}s "
            f"(dim={self._model_info['dimension']})"
        )

    @property
    def dimension(self) -> int:
        return self._model_info["dimension"]

    @property
    def model_name(self) -> str:
        return self._model_info["hf_name"]

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 8,
        use_parallel: bool = False
    ) -> List[List[float]]:
        """Embed a list of documents (code pages)."""
        self._load_model()

        # Truncate texts to avoid OOM on long code pages.
        # Approximate max chars as max_tokens * 4 (avg chars per token).
        max_chars = self._model_info.get("max_tokens", 8192) * 4
        texts = [t[:max_chars] for t in texts]

        prefix = self._model_info.get("document_prefix", "")
        if prefix:
            texts = [prefix + t for t in texts]

        if use_parallel and self._model.device.type == "cpu":
            # Multi-process encoding for CPU speedup
            logger.info(f"Starting multi-process pool for {len(texts)} documents...")
            pool = self._model.start_multi_process_pool()
            embeddings = self._model.encode_multi_process(
                texts,
                pool,
                batch_size=batch_size,
            )
            self._model.stop_multi_process_pool(pool)
        else:
            # Sequential encoding (standard)
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
                truncate_dim=self._model_info["dimension"],
            )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        self._load_model()

        prefix = self._model_info.get("query_prefix", "")
        text = prefix + query if prefix else query

        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


# ─── ChromaDB + HuggingFace Integration ─────────────────────────────────────

try:
    from chromadb import EmbeddingFunction
except ImportError:
    EmbeddingFunction = object

class ChromaHFEmbeddingFunction(EmbeddingFunction):
    """
    Adapter that plugs our HuggingFaceEmbedder into ChromaDB's
    embedding_function interface.
    """

    def __init__(self, embedder: HuggingFaceEmbedder):
        self._embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embedder.embed_documents(input)


# ─── Vector Index ────────────────────────────────────────────────────────────

class VectorCodeIndex:
    """
    Vector embedding index for code pages.

    Uses state-of-the-art open-source models from HuggingFace:
      • nomic-ai/nomic-embed-code  (default — best for code)
      • BAAI/bge-m3                (hybrid-ready alternative)
      • all-MiniLM-L6-v2           (lightweight fallback)

    Storage: ChromaDB with persistent HNSW index on disk.
    All computation is local — no API keys or cloud services needed.
    """

    def __init__(
        self,
        collection_name: str = "code_pages",
        persist_dir: str = "./data/vector_store",
        model_key: str = DEFAULT_MODEL,
    ):
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._model_key = model_key
        self._embedder = HuggingFaceEmbedder(model_key)
        self._collection = None
        self._pages_map: Dict[str, CodePage] = {}
        self._initialized = False

    def _init_chromadb(self):
        """Initialize ChromaDB client and collection with our HF embedder."""
        if self._initialized:
            return

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_dir)

            # Create embedding function adapter
            embed_fn = ChromaHFEmbeddingFunction(self._embedder)

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embed_fn,
            )
            self._initialized = True
            logger.info(
                f"ChromaDB initialized with model: {self._embedder.model_name}"
            )

        except ImportError:
            logger.warning("ChromaDB not installed — vector index disabled")
            self._initialized = False
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            self._initialized = False

    def build(self, pages: List[CodePage], batch_size: int = 8, use_parallel: bool = False):
        """
        Build the vector index from code pages.

        Embeds each page's searchable_text using the configured HuggingFace model,
        then stores embeddings + metadata in ChromaDB.
        """
        self._init_chromadb()
        if not self._initialized:
            logger.warning("Vector index not available, skipping build")
            return

        # Store page lookup
        for page in pages:
            self._pages_map[page.page_id] = page

        # Clear existing collection
        try:
            existing = self._collection.count()
            if existing > 0:
                all_ids = self._collection.get()["ids"]
                if all_ids:
                    self._collection.delete(ids=all_ids)
        except Exception:
            pass

        # Pre-compute embeddings with the HF model (for progress tracking)
        logger.info(
            f"Embedding {len(pages)} pages with {self._embedder.model_name}..."
        )
        start = time.perf_counter()

        all_texts = [p.searchable_text for p in pages]
        all_embeddings = self._embedder.embed_documents(
            all_texts,
            batch_size=batch_size,
            use_parallel=use_parallel
        )

        embed_time = time.perf_counter() - start
        logger.info(f"Embedding complete in {embed_time:.1f}s")

        # Add to ChromaDB in batches (with pre-computed embeddings)
        for i in range(0, len(pages), batch_size):
            batch_pages = pages[i:i + batch_size]
            batch_embeddings = all_embeddings[i:i + batch_size]

            self._collection.add(
                ids=[p.page_id for p in batch_pages],
                embeddings=batch_embeddings,
                documents=[p.searchable_text for p in batch_pages],
                metadatas=[
                    {
                        "symbol_name": p.symbol_name,
                        "symbol_type": p.symbol_type,
                        "file_path": p.file_path,
                        "line_start": p.line_start,
                        "line_end": p.line_end,
                    }
                    for p in batch_pages
                ],
            )

        logger.info(
            f"Built vector index: {len(pages)} pages, "
            f"model={self._embedder.model_name}, "
            f"dim={self._embedder.dimension}"
        )

    def search(self, query: str, top_k: int = 10) -> List[ScoredPage]:
        """
        Semantic similarity search using the HuggingFace embedding model.

        Embeds the query with the same model used for indexing, then
        searches ChromaDB for nearest neighbors by cosine similarity.
        """
        if not self._initialized:
            self._init_chromadb()
        if not self._initialized or self._collection is None:
            return []

        try:
            # Embed query with our HF model
            query_embedding = self._embedder.embed_query(query)

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(self._collection.count(), 1)),
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        scored_pages = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for page_id, distance in zip(ids, distances):
                page = self._pages_map.get(page_id)
                if page:
                    # ChromaDB returns cosine distances; convert to similarity
                    similarity = max(0.0, 1.0 - distance)
                    scored_pages.append(ScoredPage(
                        page=page,
                        score=similarity,
                        source="vector",
                    ))

        return scored_pages

    def get_scores(self, query: str, top_k: int = 100) -> List[tuple]:
        """Get vector similarity scores for all pages (for hybrid fusion)."""
        results = self.search(query, top_k=top_k)
        return [(sp.page.page_id, sp.score) for sp in results]

    @property
    def page_count(self) -> int:
        if self._initialized and self._collection:
            return self._collection.count()
        return 0

    @property
    def model_info(self) -> dict:
        """Return info about the active embedding model."""
        return {
            "key": self._model_key,
            "hf_name": self._embedder.model_name,
            "dimension": self._embedder.dimension,
            **EMBEDDING_MODELS[self._model_key],
        }
