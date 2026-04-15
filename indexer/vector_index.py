import logging
import time
import os
import json
import pickle
import threading
from typing import List, Dict, Optional
from dataclasses import asdict

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor
import requests
from indexer.page_index import CodePage
from indexer.bm25_index import ScoredPage
from config.settings import DEFAULT_CONFIG, OLLAMA_HOST

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    from indexer.fallback_vector_index import NumpyVectorIndex

class VectorCodeIndex:
    """
    Unified Vector Index. Prefers ChromaDB, falls back to Numpy.
    Thread-safe for parallel embedding builds.
    """

    def __init__(
        self,
        collection_name: str = "code_pages",
        persist_dir: str = "./data/vector_store",
        model_name: str = None,
    ):
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._model_name = model_name or DEFAULT_CONFIG.embedding.model_name
        self._impl = None
        self._pages_map: Dict[str, CodePage] = {}
        self._map_path = os.path.join(persist_dir, f"{collection_name}_map.json")
        self._write_lock = threading.Lock()  # Thread-safe ChromaDB writes
        self._http = requests.Session()

        if CHROMA_AVAILABLE:
            try:
                self._init_chromadb()
                if os.path.exists(self._map_path):
                    self._load_map()
            except Exception as e:
                logger.warning(f"ChromaDB hardware/binary error: {e}. Falling back to Numpy.")
                self._impl = NumpyVectorIndex(collection_name, persist_dir, model_name)
        else:
            logger.info("ChromaDB not installed. Using Numpy fallback.")
            self._impl = NumpyVectorIndex(collection_name, persist_dir, model_name)

    def _init_chromadb(self):
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings",
            model_name=self._model_name,
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_fn,
        )

    def _save_map(self):
        os.makedirs(os.path.dirname(self._map_path), exist_ok=True)
        serializable = {pid: asdict(page) for pid, page in self._pages_map.items()}
        with open(self._map_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    def _load_map(self):
        try:
            with open(self._map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._pages_map = {pid: CodePage(**page) for pid, page in raw.items()}
        except Exception as e:
            old_map_path = self._map_path.replace(".json", ".pkl")
            if os.path.exists(old_map_path):
                try:
                    with open(old_map_path, "rb") as f:
                        self._pages_map = pickle.load(f)
                    return
                except Exception:
                    pass
            logger.warning(f"Failed to load page map: {e}")

    def _get_batch_embeddings(self, texts: List[str], timeout: int = 180) -> List[List[float]]:
        """Fetch embeddings for a batch of texts from Ollama."""
        url = f"{OLLAMA_HOST}/api/embed"
        payload = {"model": self._model_name, "input": texts}
        response = self._http.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["embeddings"]

    def _safe_embed_batch(self, batch_pages: List[CodePage], current_batch_size: int) -> List[List[float]]:
        """
        Recursively attempts to embed a batch, scaling down on failure.
        """
        texts = [p.searchable_text[:6000] for p in batch_pages]

        try:
            return self._get_batch_embeddings(texts)
        except (requests.exceptions.RequestException, Exception) as e:
            if current_batch_size <= 1:
                logger.error(f"Atomic embedding failure for {batch_pages[0].page_id}: {e}")
                return [[0.0] * DEFAULT_CONFIG.embedding.dimension]

            new_size = max(1, current_batch_size // 2)
            logger.warning(f"Ollama pressure detected. Scaling down: {current_batch_size} -> {new_size}")

            # Split and recurse
            mid = len(batch_pages) // 2
            left_embs = self._safe_embed_batch(batch_pages[:mid], new_size)
            right_embs = self._safe_embed_batch(batch_pages[mid:], new_size)

            return left_embs + right_embs

    def build(self, pages: List[CodePage], batch_size: int = None):
        if self._impl:
            return self._impl.build(pages, batch_size)

        cfg = DEFAULT_CONFIG.embedding
        batch_size = batch_size or cfg.batch_size
        max_workers = cfg.max_workers

        # Silence noisy HTTP logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        # Clear existing collection efficiently — delete + recreate instead of
        # fetching all IDs into memory (which can OOM on 100K+ pages)
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embed_fn,
            )
        except Exception as e:
            logger.warning(f"Failed to clear collection: {e}")

        self._pages_map = {p.page_id: p for p in pages}
        self._save_map()

        start = time.perf_counter()

        # Prepare batches
        batches = []
        for i in range(0, len(pages), batch_size):
            batches.append(pages[i:i + batch_size])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=logging.getLogger("code_graph_rag").handlers[0].console if logging.getLogger("code_graph_rag").handlers else None
        ) as progress:
            task = progress.add_task(f"[cyan]Embedding {len(pages)} pages (Turbo Mode, {max_workers} threads)...", total=len(pages))

            def process_batch(batch_pages):
                embeddings = self._safe_embed_batch(batch_pages, len(batch_pages))
                texts = [p.searchable_text[:6000] for p in batch_pages]

                # Thread-safe ChromaDB write — PersistentClient is NOT safe for
                # concurrent writes, so we serialize add() calls with a lock.
                with self._write_lock:
                    self._collection.add(
                        ids=[p.page_id for p in batch_pages],
                        embeddings=embeddings,
                        documents=texts,
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
                progress.update(task, advance=len(batch_pages))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_batch, batches)

        logger.info(f"Built Chroma vector index (Parallel) in {time.perf_counter() - start:.1f}s")

    def search(self, query: str, top_k: int = 10) -> List[ScoredPage]:
        if self._impl:
            return self._impl.search(query, top_k)

        try:
            # Truncate query too
            results = self._collection.query(query_texts=[query[:6000]], n_results=top_k)
            scored_pages = []

            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

                for pid, dist in zip(ids, distances):
                    page = self._pages_map.get(pid)
                    if page:
                        similarity = max(0.0, 1.0 - dist)
                        scored_pages.append(ScoredPage(
                            page=page,
                            score=float(similarity),
                            source="vector_chroma"
                        ))
            return scored_pages
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_scores(self, query: str, top_k: int = 100) -> List[tuple]:
        """Get vector similarity scores for all pages (for hybrid fusion)."""
        results = self.search(query, top_k=top_k)
        return [(sp.page.page_id, sp.score) for sp in results]

    @property
    def page_count(self) -> int:
        if self._impl: return self._impl.page_count
        return self._collection.count() if self._collection else 0
