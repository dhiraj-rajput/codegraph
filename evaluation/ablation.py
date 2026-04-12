"""
Ablation study runner — systematically tests component contributions.

Runs the 8-experiment ablation suite across all retrieval configurations
and produces comparison tables for the paper.

Experiment configurations:
  E1: BM25 only
  E2: TF-IDF only
  E3: BM25 + Symbol Index
  E4: BM25 + Graph Traversal
  E5: BM25 + Symbol + Graph (full vectorless)
  E6: Vector only (DPR / embedding)
  E7: Vector + Graph
  E8: Hybrid (BM25 + Vector + Symbol + Graph)
"""

import json
import time
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from retriever.base_retriever import BaseRetriever, RetrievalResult
from evaluation.metrics import compute_all_metrics, EvaluationResult
from evaluation.efficiency import EfficiencyTracker
from evaluation.statistics import paired_t_test, bootstrap_test, StatTestResult

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
    efficiency: dict
    per_query_results: List[dict] = field(default_factory=list)


class AblationRunner:
    """
    Runs systematic ablation studies across retrieval configurations.

    Usage:
        runner = AblationRunner()
        runner.add_method("BM25 only", bm25_retriever)
        runner.add_method("Vectorless", vectorless_retriever)
        results = runner.run(benchmarks)
        runner.print_table(results)
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

        Returns a list of AblationResult, one per method.
        """
        results = []

        for method_name, retriever in self._methods.items():
            logger.info(f"Running ablation: {method_name}")
            tracker = EfficiencyTracker(method_name)

            retrieved_lists = []
            relevant_sets = []
            latencies = []
            per_query = []

            for bench in benchmarks:
                # Build relevant set from both files and symbols
                relevant = set(bench.relevant_files + bench.relevant_symbols)

                with tracker.measure_retrieval():
                    result = retriever.retrieve(bench.query, top_k=top_k)

                # Collect retrieved identifiers (file paths and symbol names)
                retrieved_ids = []
                for sp in result.pages:
                    retrieved_ids.append(sp.page.file_path)
                    retrieved_ids.append(sp.page.symbol_name)
                    retrieved_ids.append(sp.page.qualified_name)

                retrieved_lists.append(retrieved_ids)
                relevant_sets.append(relevant)
                latencies.append(result.retrieval_time_ms)

                per_query.append({
                    "query_id": bench.query_id,
                    "query": bench.query,
                    "retrieved_files": result.file_paths[:top_k],
                    "latency_ms": result.retrieval_time_ms,
                })

            # Compute all metrics
            evaluation = compute_all_metrics(
                retrieved_lists=retrieved_lists,
                relevant_sets=relevant_sets,
                latencies=latencies,
                method_name=method_name,
            )

            efficiency_report = tracker.report()

            results.append(AblationResult(
                experiment_id=method_name.lower().replace(" ", "_").replace("+", "_"),
                method_name=method_name,
                components=self._method_components[method_name],
                evaluation=evaluation,
                efficiency=efficiency_report.to_dict(),
                per_query_results=per_query,
            ))

        return results

    def run_statistical_tests(
        self,
        results: List[AblationResult],
        metric: str = "recall@5",
    ) -> List[StatTestResult]:
        """
        Run pairwise statistical significance tests.

        Compares every pair of methods on the specified metric.
        """
        # Build per-query score maps
        score_map = {}
        for r in results:
            if metric == "recall@5":
                score_map[r.method_name] = r.evaluation.per_query_recall_5
            elif metric == "mrr":
                score_map[r.method_name] = r.evaluation.per_query_mrr
            elif metric == "f1":
                score_map[r.method_name] = r.evaluation.per_query_f1

        from evaluation.statistics import compare_methods
        return compare_methods(score_map, metric=metric)

    @staticmethod
    def print_table(results: List[AblationResult]):
        """Print a formatted comparison table."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Ablation Study Results", show_lines=True)

            table.add_column("Method", style="cyan", width=25)
            table.add_column("Recall@1", justify="right")
            table.add_column("Recall@5", justify="right")
            table.add_column("Recall@10", justify="right")
            table.add_column("MRR", justify="right")
            table.add_column("nDCG@5", justify="right")
            table.add_column("Precision@5", justify="right")
            table.add_column("Hit@5", justify="right")
            table.add_column("Latency", justify="right")

            for r in results:
                e = r.evaluation
                table.add_row(
                    r.method_name,
                    f"{e.recall_at_1:.3f}",
                    f"{e.recall_at_5:.3f}",
                    f"{e.recall_at_10:.3f}",
                    f"{e.mrr_score:.3f}",
                    f"{e.ndcg_at_5:.3f}",
                    f"{e.precision_at_5:.3f}",
                    f"{e.hit_at_5:.3f}",
                    f"{e.avg_latency_ms:.1f}ms",
                )

            console.print(table)

        except ImportError:
            # Fallback: plain text
            header = f"{'Method':<25} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@5':>7} {'Lat':>8}"
            print(header)
            print("-" * len(header))
            for r in results:
                e = r.evaluation
                print(
                    f"{r.method_name:<25} {e.recall_at_1:>6.3f} {e.recall_at_5:>6.3f} "
                    f"{e.recall_at_10:>6.3f} {e.mrr_score:>6.3f} {e.ndcg_at_5:>7.3f} "
                    f"{e.avg_latency_ms:>7.1f}ms"
                )

    @staticmethod
    def save_results(results: List[AblationResult], path: str):
        """Save ablation results to JSON."""
        data = []
        for r in results:
            data.append({
                "experiment_id": r.experiment_id,
                "method_name": r.method_name,
                "components": r.components,
                "metrics": r.evaluation.to_dict(),
                "efficiency": r.efficiency,
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved ablation results to {path}")
