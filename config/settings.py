"""
Global configuration for the Code Graph RAG system.
All paths and Ollama model settings are centralized here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
GRAPH_DIR = PROJECT_ROOT / "data" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for d in [DATA_DIR, INDEX_DIR, GRAPH_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Keys & Environment ──────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ─── Model Configuration ────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """
    Configuration for local Ollama embedding models.
    """
    model_name: str = "nomic-embed-text-v2-moe"     # Standard fast reliable code/text embedding
    dimension: int = 768
    batch_size: int = 100
    max_workers: int = os.cpu_count() or 4   # Dynamic workers based on CPU cores

    
@dataclass
class LLMConfig:
    """Configuration for local Ollama LLM inference."""
    provider: str = "ollama"
    model_name: str = "llama3.2"             # Fast and excellent general purpose/code reasoning
    fast_model_name: str = "llama3.2"        # Quick rewrites etc.
    
    enable_query_expansion: bool = True      # Rewrites vague queries
    temperature: float = 0.0
    max_tokens: int = 2048
    token_budget: int = 4000                 # Max context tokens for retrieval

@dataclass
class RetrieverConfig:
    """Configuration for retrieval pipelines."""
    top_k: int = 10
    bm25_weight: float = 0.4
    symbol_weight: float = 0.3
    graph_weight: float = 0.3
    vector_weight: float = 0.5
    graph_depth: int = 2
    # Hybrid weights
    hybrid_bm25_weight: float = 0.5
    hybrid_vector_weight: float = 0.3
    hybrid_graph_weight: float = 0.2

@dataclass
class ParserConfig:
    """Configuration for code parsing."""
    supported_extensions: tuple = (".py", ".js", ".ts", ".go", ".java", ".cpp", ".rs")
    max_file_size_bytes: int = 1_000_000      # Skip files > 1MB
    max_function_tokens: int = 512            # Split large functions
    ignore_patterns: tuple = (
        "__pycache__", ".git", "node_modules", ".tox", ".eggs",
        "*.pyc", "*.egg-info", "dist", "build", ".venv", "venv",
    )

@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)

# ─── Default Config Instance ────────────────────────────────────────────────

DEFAULT_CONFIG = ExperimentConfig()
