import json
import time
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from retriever.base_retriever import BaseRetriever
from evaluation.metrics import compute_all_metrics, EvaluationResult

logger = logging.getLogger(__name__)

@dataclass
class QueryBenchmark:
    """A single benchmark query with ground truth."""
    query_id: str
    query: str
    category: str                     # architecture | dependency | debugging
    relevant_files: List[str]         # Ground truth files
    relevant_symbols: List[str]       # Ground truth symbol names
    ground_truth_answer: str = ""      # Expected answer text
    repository: str = ""

@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    experiment_id: str
    method_name: str
    components: List[str]              # Active components
    evaluation: EvaluationResult
    per_query_results: List[dict] = field(default_factory=list)

class AblationRunner:
    """
    Runs systematic ablation studies across retrieval configurations.
    """

    def __init__(self):
        self._methods: Dict[str, BaseRetriever] = {}
        self._method_components: Dict[str, List[str]] = {}

    def add_method(
        self, name: str, retriever: BaseRetriever, components: Optional[List[str]] = None
    ):
        """Register a retrieval method for testing."""
        self._methods[name] = retriever
        self._method_components[name] = components or [name]

    def run(
        self,
        benchmarks: List[QueryBenchmark],
        top_k: int = 10,
    ) -> List[AblationResult]:
        """
        Run all registered methods on all benchmarks.
        """
        results = []

        for method_name, retriever in self._methods.items():
            logger.info(f"Running ablation: {method_name}")
            
            retrieved_lists = []
            relevant_sets = []
            latencies = []
            per_query = []

            for bench in benchmarks:
                # Build relevant set from both files and symbols
                relevant = set(bench.relevant_files + bench.relevant_symbols)

                start_time = time.perf_counter()
                result = retriever.retrieve(bench.query, top_k=top_k)
                latency = (time.perf_counter() - start_time) * 1000

                # Collect retrieved identifiers (file paths and symbol names)
                retrieved_ids = []
                for sp in result.pages:
                    retrieved_ids.append(sp.page.file_path)
                    retrieved_ids.append(sp.page.symbol_name)
                    # Filter out anything not containing relevant substrings
                    # Actually standard recall is strict on ID match
                
                retrieved_lists.append(retrieved_ids)
                relevant_sets.append(relevant)
                latencies.append(latency)

                per_query.append({
                    "query_id": bench.query_id,
                    "query": bench.query,
                    "latency_ms": latency,
                })

            # Compute all metrics
            evaluation = compute_all_metrics(
                retrieved_lists=retrieved_lists,
                relevant_sets=relevant_sets,
                latencies=latencies,
                method_name=method_name,
            )

            results.append(AblationResult(
                experiment_id=method_name.lower().replace(" ", "_").replace("+", "_"),
                method_name=method_name,
                components=self._method_components[method_name],
                evaluation=evaluation,
                per_query_results=per_query,
            ))

        return results

    @staticmethod
    def print_table(results: List[AblationResult]):
        """Print a formatted comparison table."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="CodeGraph RAG — Benchmark Results", show_lines=True)

        table.add_column("Retrieval Strategy", style="cyan", width=25)
        table.add_column("Recall@5", justify="right")
        table.add_column("Recall@10", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("p95 Latency", justify="right")

        for r in results:
            e = r.evaluation
            table.add_row(
                r.method_name,
                f"{e.recall_at_5:.3f}",
                f"{e.recall_at_10:.3f}",
                f"{e.mrr_score:.3f}",
                f"{e.latency_ms:.1f}ms",
                f"{e.p95_latency_ms:.1f}ms",
            )

        console.print(table)
