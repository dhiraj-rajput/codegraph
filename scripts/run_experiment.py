"""
Experiment runner — automates the full ablation study pipeline.

Usage:
    python scripts/run_experiment.py --repo <repo_path> --name <index_name>
    python scripts/run_experiment.py --index <index_name> --queries datasets/queries/sample.json

This script:
1. Ingests a repository (if --repo provided)
2. Builds all indexes (BM25, Symbol, Vector, Graph)
3. Creates all retriever configurations for ablation
4. Runs retrieval on all benchmark queries
5. Computes all IR + QA metrics
6. Runs statistical significance tests
7. Saves results and prints comparison table
"""

import sys
import json
import logging
import time
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import click
from rich.console import Console
from rich.logging import RichHandler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DEFAULT_CONFIG, INDEX_DIR, RESULTS_DIR
from parser.tree_sitter_parser import CodeParser
from parser.symbol_extractor import SymbolExtractor
from graph_builder.code_graph import CodeGraph
from indexer.bm25_index import BM25CodeIndex
from indexer.symbol_index import SymbolIndex
from indexer.page_index import PageIndex
from indexer.vector_index import VectorCodeIndex
from retriever.vector_retriever import VectorRetriever
from retriever.vectorless_retriever import VectorlessRetriever
from retriever.hybrid_retriever import HybridRetriever
from evaluation.ablation import AblationRunner, QueryBenchmark
from evaluation.efficiency import EfficiencyTracker
from evaluation.statistics import paired_t_test, bootstrap_test

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)
logger = logging.getLogger(__name__)


def load_benchmarks(path: str) -> list:
    """Load query benchmarks from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    benchmarks = []
    for item in data:
        benchmarks.append(QueryBenchmark(
            query_id=item["query_id"],
            query=item["query"],
            category=item.get("category", "general"),
            relevant_files=item.get("relevant_files", []),
            relevant_symbols=item.get("relevant_symbols", []),
            ground_truth_answer=item.get("ground_truth_answer", ""),
            repository=item.get("repository", ""),
        ))
    return benchmarks


def ingest_repository(repo_path: str, index_name: str, model_key: str = "nomic-code", parallel: bool = False):
    """Parse and index a repository."""
    console.print(f"\n[bold cyan]═══ Ingesting: {repo_path} ═══[/bold cyan]\n")

    tracker = EfficiencyTracker("ingest")

    with tracker.measure_build():
        # Parse
        parser = CodeParser()
        parsed_files = parser.parse_repository(repo_path)
        console.print(f"  [green]✓[/green] Parsed {len(parsed_files)} files")

        # Extract symbols
        extractor = SymbolExtractor()
        symbols = extractor.extract_from_repository(parsed_files)
        console.print(f"  [green]✓[/green] Extracted {len(symbols)} symbols")

        # Build graph
        graph = CodeGraph()
        graph.build(symbols)
        stats = graph.stats()
        console.print(f"  [green]✓[/green] Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # Build all indexes
        page_index = PageIndex()
        page_index.build(symbols)
        page_index.build_from_files(parsed_files)
        console.print(f"  [green]✓[/green] Page index: {page_index.page_count} pages")

        bm25_index = BM25CodeIndex()
        bm25_index.build(page_index.all_pages)
        console.print(f"  [green]✓[/green] BM25 index: {bm25_index.page_count} documents")

        sym_index = SymbolIndex()
        sym_index.build(symbols)
        console.print(f"  [green]✓[/green] Symbol index: {sym_index.count} entries")

        vec_index = VectorCodeIndex(
            collection_name=index_name,
            persist_dir=str(INDEX_DIR / "vector_store"),
            model_key=model_key,
        )
        console.print(f"  Embedding model: [cyan]{vec_index.model_info['hf_name']}[/cyan]")
        vec_index.build(page_index.all_pages, use_parallel=parallel)
        console.print(f"  [green]✓[/green] Vector index: {vec_index.page_count} embeddings")

        # Save
        save_dir = INDEX_DIR / index_name
        save_dir.mkdir(parents=True, exist_ok=True)
        graph.save(str(save_dir / "graph.pkl"))
        bm25_index.save(str(save_dir / "bm25.pkl"))
        sym_index.save(str(save_dir / "symbols.db"))

    report = tracker.report()
    console.print(f"\n  Build time: {report.index_build_time_sec:.1f}s | "
                  f"Peak memory: {report.peak_memory_mb:.1f} MB\n")

    return graph, bm25_index, sym_index, page_index, vec_index


def load_indexes(index_name: str, model_key: str = "nomic-code"):
    """Load pre-built indexes from disk."""
    save_dir = INDEX_DIR / index_name
    if not save_dir.exists():
        console.print(f"[red]Index not found: {save_dir}[/red]")
        sys.exit(1)

    graph = CodeGraph()
    graph.load(str(save_dir / "graph.pkl"))

    bm25_index = BM25CodeIndex()
    bm25_index.load(str(save_dir / "bm25.pkl"))

    sym_index = SymbolIndex()
    sym_index.load(str(save_dir / "symbols.db"))

    page_index = PageIndex()
    page_index.build(graph.all_symbols)

    vec_index = VectorCodeIndex(
        collection_name=index_name,
        persist_dir=str(INDEX_DIR / "vector_store"),
        model_key=model_key,
    )

    return graph, bm25_index, sym_index, page_index, vec_index


def build_ablation_methods(graph, bm25_index, sym_index, page_index, vec_index):
    """Create all ablation retriever configurations."""
    runner = AblationRunner()

    # ── E1: BM25 only ──
    # Uses a vectorless retriever with only BM25 weight
    from config.settings import RetrieverConfig
    bm25_only_cfg = RetrieverConfig(
        bm25_weight=1.0, symbol_weight=0.0, graph_weight=0.0, graph_depth=0
    )
    bm25_only = VectorlessRetriever(
        bm25_index=bm25_index, symbol_index=sym_index,
        page_index=page_index, code_graph=graph, config=bm25_only_cfg,
    )
    runner.add_method("BM25 only", bm25_only, ["BM25"])

    # ── E2: BM25 + Symbol ──
    bm25_sym_cfg = RetrieverConfig(
        bm25_weight=0.6, symbol_weight=0.4, graph_weight=0.0, graph_depth=0
    )
    bm25_sym = VectorlessRetriever(
        bm25_index=bm25_index, symbol_index=sym_index,
        page_index=page_index, code_graph=graph, config=bm25_sym_cfg,
    )
    runner.add_method("BM25 + Symbol", bm25_sym, ["BM25", "Symbol"])

    # ── E3: BM25 + Graph ──
    bm25_graph_cfg = RetrieverConfig(
        bm25_weight=0.6, symbol_weight=0.0, graph_weight=0.4, graph_depth=2
    )
    bm25_graph = VectorlessRetriever(
        bm25_index=bm25_index, symbol_index=sym_index,
        page_index=page_index, code_graph=graph, config=bm25_graph_cfg,
    )
    runner.add_method("BM25 + Graph", bm25_graph, ["BM25", "Graph"])

    # ── E4: BM25 + Symbol + Graph (full vectorless = System B) ──
    vectorless = VectorlessRetriever(
        bm25_index=bm25_index, symbol_index=sym_index,
        page_index=page_index, code_graph=graph,
    )
    runner.add_method("Vectorless (full)", vectorless, ["BM25", "Symbol", "Graph"])

    # ── E5: Vector only (System A) ──
    vector_only = VectorRetriever(vector_index=vec_index)
    runner.add_method("Vector only", vector_only, ["Vector"])

    # ── E6: Hybrid (System C) ──
    hybrid = HybridRetriever(
        bm25_index=bm25_index, vector_index=vec_index,
        symbol_index=sym_index, page_index=page_index, code_graph=graph,
    )
    runner.add_method("Hybrid (full)", hybrid, ["BM25", "Vector", "Symbol", "Graph"])

    return runner


@click.command()
@click.option("--repo", default=None, help="Path to repository to ingest")
@click.option("--name", default="experiment", help="Index name")
@click.option("--index", default=None, help="Use existing index (skip ingest)")
@click.option("--queries", default=None, help="Path to benchmark queries JSON")
@click.option("--top-k", default=10, help="Top-K for retrieval")
@click.option("--output", default=None, help="Output path for results JSON")
@click.option("--swe-bench", is_flag=True, help="Use SWE-bench FastAPI subset")
@click.option("--parallel", is_flag=True, help="Use multi-core parallel embedding (CPU speedup)")
@click.option("--fast", is_flag=True, help="Fast mode: use lightweight model + parallel embedding")
@click.option("--model", default="nomic-code",
              type=click.Choice(["nomic-code", "bge-m3", "nomic-text", "bge-base", "minilm"]),
              help="HuggingFace embedding model for vector baseline")
def main(repo, name, index, queries, top_k, output, swe_bench, parallel, fast, model):
    """Run the full ablation experiment suite."""
    if fast:
        model = "minilm"
        parallel = True


    console.print("[bold magenta]╔══════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   Code Graph RAG — Experiment Runner ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════╝[/bold magenta]")

    # Step 1: Ingest or load
    if repo:
        graph, bm25_index, sym_index, page_index, vec_index = ingest_repository(
            repo, name, model_key=model, parallel=parallel
        )
    elif index:
        console.print(f"\n[dim]Loading existing index: {index}[/dim]")
        graph, bm25_index, sym_index, page_index, vec_index = load_indexes(index, model_key=model)
    else:
        console.print("[red]Provide --repo or --index[/red]")
        sys.exit(1)

    # Step 2: Load or generate benchmarks
    if queries:
        benchmarks = load_benchmarks(queries)
        console.print(f"\n[green]✓[/green] Loaded {len(benchmarks)} benchmark queries")
    elif swe_bench:
        from datasets.swe_bench import load_swe_bench_fastapi_subset
        benchmarks = load_swe_bench_fastapi_subset()
        console.print(f"\n[green]✓[/green] Loaded {len(benchmarks)} SWE-Bench FastAPI issues")
    else:
        # Generate simple benchmarks from the graph itself
        console.print("\n[yellow]No --queries provided, generating synthetic benchmarks...[/yellow]")
        benchmarks = _generate_synthetic_benchmarks(graph)
        console.print(f"[green]✓[/green] Generated {len(benchmarks)} synthetic queries")

    # Step 3: Build all ablation methods
    console.print("\n[bold]Setting up ablation methods...[/bold]")
    runner = build_ablation_methods(graph, bm25_index, sym_index, page_index, vec_index)

    # Step 4: Run ablation
    console.print("\n[bold cyan]═══ Running Ablation Studies ═══[/bold cyan]\n")
    results = runner.run(benchmarks, top_k=top_k)

    # Step 5: Print results table
    console.print("\n")
    runner.print_table(results)

    # Step 6: Statistical significance tests
    console.print("\n[bold cyan]═══ Statistical Significance ═══[/bold cyan]\n")
    stat_tests = runner.run_statistical_tests(results, metric="recall@5")
    for test in stat_tests:
        sig_marker = "[green]✓ SIG[/green]" if test.significant else "[dim]n.s.[/dim]"
        console.print(
            f"  {test.method_a} vs {test.method_b}: "
            f"p={test.p_value:.4f} {sig_marker} | d={test.effect_size:.3f}"
        )

    # Step 7: Save results
    out_path = output or str(RESULTS_DIR / f"ablation_{name}_{int(time.time())}.json")
    runner.save_results(results, out_path)
    console.print(f"\n[green]✓ Results saved to {out_path}[/green]")


def _generate_synthetic_benchmarks(graph: CodeGraph) -> list:
    """Generate simple benchmarks from graph symbols (when no query file is given)."""
    benchmarks = []
    symbols = graph.all_symbols

    # Take up to 30 functions/classes for synthetic queries
    candidates = [s for s in symbols if s.type in ("function", "class", "method")][:30]

    for i, sym in enumerate(candidates):
        # Architecture-style query
        benchmarks.append(QueryBenchmark(
            query_id=f"syn_arch_{i}",
            query=f"How does {sym.name} work?",
            category="architecture",
            relevant_files=[sym.file_path],
            relevant_symbols=[sym.name, sym.qualified_name],
        ))

        # Dependency-style query
        if sym.calls:
            benchmarks.append(QueryBenchmark(
                query_id=f"syn_dep_{i}",
                query=f"What does {sym.name} call?",
                category="dependency",
                relevant_files=[sym.file_path],
                relevant_symbols=[sym.name] + sym.calls[:3],
            ))

    return benchmarks


if __name__ == "__main__":
    main()
