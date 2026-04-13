"""
Simple Numpy-based vector index for environments where ChromaDB/C-builds fail.
Uses cosine similarity and stores everything in a pickle file.
"""

import os
import pickle
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from indexer.page_index import CodePage
from indexer.bm25_index import ScoredPage
from config.settings import DEFAULT_CONFIG, OLLAMA_HOST
from concurrent.futures import ThreadPoolExecutor
import requests
import urllib.request
import json
import time

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)

class NumpyVectorIndex:
    """
    Zero-dependency (except numpy) vector index.
    Fetches embeddings from Ollama and performs search using Numpy.
    """
    def __init__(
        self,
        collection_name: str = "code_pages",
        persist_dir: str = "./data/vector_store",
        model_name: str = None,
    ):
        self._name = collection_name
        self._persist_dir = persist_dir
        self._model_name = model_name or DEFAULT_CONFIG.embedding.model_name
        self._save_path = os.path.join(persist_dir, f"{collection_name}_numpy.pkl")
        
        self.embeddings: Optional[np.ndarray] = None
        self.page_ids: List[str] = []
        self.pages_map: Dict[str, CodePage] = {}
        
        if os.path.exists(self._save_path):
            self.load()

    def _get_batch_embeddings(self, texts: List[str], timeout: int = 180) -> List[List[float]]:
        """Fetch embeddings for a batch of texts from Ollama."""
        url = f"{OLLAMA_HOST}/api/embed"
        payload = {"model": self._model_name, "input": texts}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["embeddings"]

    def _safe_embed_batch(self, batch_pages: List[CodePage], current_batch_size: int) -> List[List[float]]:
        """Recursive safety for Numpy index."""
        texts = [p.searchable_text[:6000] for p in batch_pages]
        try:
            return self._get_batch_embeddings(texts)
        except Exception as e:
            if current_batch_size <= 1:
                logger.error(f"Atomic embedding failure: {e}")
                return [[0.0] * 768]
            
            new_size = max(1, current_batch_size // 2)
            logger.warning(f"Ollama pressure (Numpy). Scaling down: {current_batch_size} -> {new_size}")
            
            mid = len(batch_pages) // 2
            return self._safe_embed_batch(batch_pages[:mid], new_size) + self._safe_embed_batch(batch_pages[mid:], new_size)

    def build(self, pages: List[CodePage], batch_size: int = None):
        cfg = DEFAULT_CONFIG.embedding
        batch_size = batch_size or cfg.batch_size
        max_workers = cfg.max_workers
        
        start = time.perf_counter()
        self.page_ids = []
        self.pages_map = {}
        embeddings_list = [None] * len(pages)
        
        # Mapping for progress tracking
        batches = []
        for i in range(0, len(pages), batch_size):
            batches.append((i, pages[i:i + batch_size]))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=logging.getLogger("code_graph_rag").handlers[0].console if logging.getLogger("code_graph_rag").handlers else None
        ) as progress:
            task = progress.add_task(f"[cyan]Building Numpy index (Turbo Mode, {max_workers} threads)...", total=len(pages))
            
            def process_batch(batch_info):
                idx, batch_pages = batch_info
                batch_embeddings = self._safe_embed_batch(batch_pages, len(batch_pages))
                
                for i, emb in enumerate(batch_embeddings):
                    embeddings_list[idx + i] = emb
                    
                progress.update(task, advance=len(batch_pages))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_batch, batches)

        # Update metadata
        for p in pages:
            self.page_ids.append(p.page_id)
            self.pages_map[p.page_id] = p
        
        self.embeddings = np.array(embeddings_list)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-9)
        
        self.save()
        logger.info(f"Built Numpy vector index (Parallel) in {time.perf_counter() - start:.1f}s")

    def search(self, query: str, top_k: int = 10) -> List[ScoredPage]:
        if self.embeddings is None:
            return []
            
        q_embs = self._get_ollama_embeddings(query)
        q_emb = np.array(q_embs[0])
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        
        # Cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            pid = self.page_ids[idx]
            results.append(ScoredPage(
                page=self.pages_map[pid],
                score=float(similarities[idx]),
                source="vector_numpy"
            ))
        return results

    def save(self):
        os.makedirs(self._persist_dir, exist_ok=True)
        with open(self._save_path, 'wb') as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "page_ids": self.page_ids,
                "pages_map": self.pages_map,
                "model": self._model_name
            }, f)

    def load(self):
        with open(self._save_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.page_ids = data["page_ids"]
            self.pages_map = data["pages_map"]
            self._model_name = data.get("model", self._model_name)

    @property
    def page_count(self) -> int:
        return len(self.page_ids)
