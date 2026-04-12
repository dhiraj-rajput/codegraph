"""
Global configuration for the Code Graph RAG system.
All paths, API keys, and model settings are centralized here.
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
for d in [DATA_DIR, INDEX_DIR, GRAPH_DIR, RESULTS_DIR,
          RESULTS_DIR / "tables", RESULTS_DIR / "plots", RESULTS_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)


# ─── API Keys (from environment) ────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    # Load .env explicitly from PROJECT_ROOT
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")


# ─── Model Configuration ────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding models.

    Model keys (map to EMBEDDING_MODELS registry in indexer/vector_index.py):

      Tier 1 — Best for code (recommended):
        "nomic-code"  →  nomic-ai/nomic-embed-code        (768d, 8192 tokens, Apache 2.0)
        "bge-m3"      →  BAAI/bge-m3                      (1024d, 8192 tokens, MIT)

      Tier 2 — Strong general-purpose:
        "nomic-text"  →  nomic-ai/nomic-embed-text-v2-moe (768d, MoE, Apache 2.0)
        "bge-base"    →  BAAI/bge-base-en-v1.5            (768d, 512 tokens, MIT)

      Tier 3 — Lightweight / CPU-only:
        "minilm"      →  sentence-transformers/all-MiniLM-L6-v2  (384d, 256 tokens)
    """
    model_key: str = "nomic-code"                # Best free code embedding model
    model_name: str = "nomic-ai/nomic-embed-code"
    dimension: int = 768
    batch_size: int = 32


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""
    provider: str = "gemini"                     # "gemini" | "huggingface" | "openai"
    model_name: str = "gemini-1.5-flash"         # Best free/cheap balanced model
    fast_model_name: str = "gemini-1.5-flash"    # Used for fast inner-loop tasks like Query Rewriting
    
    # Hugging Face fallback (Llama 3 recommended)
    hf_model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    
    enable_query_expansion: bool = True          # Rewrites vague queries with LLM
    temperature: float = 0.0
    max_tokens: int = 2048
    token_budget: int = 4000                     # Max context tokens for retrieval


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
    max_file_size_bytes: int = 1_000_000          # Skip files > 1MB
    max_function_tokens: int = 512                # Split large functions
    ignore_patterns: tuple = (
        "__pycache__", ".git", "node_modules", ".tox", ".eggs",
        "*.pyc", "*.egg-info", "dist", "build", ".venv", "venv",
    )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    recall_k_values: tuple = (1, 5, 10)
    precision_k_values: tuple = (5, 10)
    ndcg_k_values: tuple = (5, 10)
    judge_model: str = "gpt-4o"
    judge_temperature: float = 0.0
    bootstrap_samples: int = 1000
    significance_level: float = 0.05


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ─── Default Config Instance ────────────────────────────────────────────────

DEFAULT_CONFIG = ExperimentConfig()
