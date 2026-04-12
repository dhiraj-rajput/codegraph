"""
Code Graph RAG — CLI Entry Point

Research-grade system for comparing Vector, Vectorless, and Hybrid RAG
on codebase retrieval tasks.

Usage:
    python main.py ingest <repo_path>
    python main.py query "How does authentication work?"
    python main.py experiment --config experiments.yaml
    python main.py ablation --benchmarks queries.json
"""

import sys
import json
import logging
import time
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from config.settings import DEFAULT_CONFIG, DATA_DIR, INDEX_DIR, GRAPH_DIR, RESULTS_DIR
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
from query_engine.context_builder import ContextBuilder
from query_engine.prompt_templates import SYSTEM_PROMPT, CONTEXT_TEMPLATE
from llm_interface.llm_client import LLMClient
from evaluation.efficiency import EfficiencyTracker

console = Console()

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)
logger = logging.getLogger("code_graph_rag")


# ─── CLI App ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug):
    """Code Graph RAG — Research-grade vectorless RAG for codebases."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


# ─── Ingest Command ─────────────────────────────────────────────────────────

@cli.command()
@click.argument("repo_path")
@click.option("--name", default=None, help="Name for this repository index")
@click.option("--vector/--no-vector", default=True, help="Build vector index")
@click.option("--model", default="nomic-code",
              type=click.Choice(["nomic-code", "bge-m3", "nomic-text", "bge-base", "minilm"]),
              help="HuggingFace embedding model (default: nomic-embed-code, best for code)")
def ingest(repo_path, name, vector, model):
    """Ingest a repository: parse → graph → index."""
    repo = Path(repo_path)
    if not repo.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {repo_path}")
        sys.exit(1)

    repo_name = name or repo.name
    console.print(f"\n[bold cyan]Ingesting repository:[/bold cyan] {repo.absolute()}")
    console.print(f"[dim]Index name: {repo_name}[/dim]\n")

    tracker = EfficiencyTracker("ingest")

    with tracker.measure_build():
        # Step 1: Parse
        console.print("[bold]Step 1/4:[/bold] Parsing source files...")
        parser = CodeParser()
        parsed_files = parser.parse_repository(str(repo))
        console.print(f"  ✓ Parsed {len(parsed_files)} files")

        # Step 2: Extract symbols
        console.print("[bold]Step 2/4:[/bold] Extracting symbols...")
        extractor = SymbolExtractor()
        symbols = extractor.extract_from_repository(parsed_files)
        console.print(f"  ✓ Extracted {len(symbols)} symbols")

        # Step 3: Build graph
        console.print("[bold]Step 3/4:[/bold] Building code graph...")
        graph = CodeGraph()
        graph.build(symbols)
        stats = graph.stats()
        console.print(f"  ✓ Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # Step 4: Build indexes
        console.print("[bold]Step 4/4:[/bold] Building indexes...")

        # Page index
        page_index = PageIndex()
        page_index.build(symbols)
        page_index.build_from_files(parsed_files)
        console.print(f"  ✓ Page index: {page_index.page_count} pages")

        # BM25 index
        bm25_index = BM25CodeIndex()
        bm25_index.build(page_index.all_pages)
        console.print(f"  ✓ BM25 index: {bm25_index.page_count} documents")

        # Symbol index
        sym_index = SymbolIndex()
        sym_index.build(symbols)
        console.print(f"  ✓ Symbol index: {sym_index.count} entries")

        # Vector index (optional)
        if vector:
            from indexer.vector_index import EMBEDDING_MODELS
            model_info = EMBEDDING_MODELS[model]
            console.print(f"  Embedding with [cyan]{model_info['hf_name']}[/cyan] ...")
            console.print(f"  [dim](dim={model_info['dimension']}, max_tokens={model_info['max_tokens']})[/dim]")
            vec_index = VectorCodeIndex(
                collection_name=repo_name,
                persist_dir=str(INDEX_DIR / "vector_store"),
                model_key=model,
            )
            vec_index.build(page_index.all_pages)
            console.print(f"  ✓ Vector index: {vec_index.page_count} embeddings")

        # Save indexes
        save_dir = INDEX_DIR / repo_name
        save_dir.mkdir(parents=True, exist_ok=True)

        graph.save(str(save_dir / "graph.pkl"))
        bm25_index.save(str(save_dir / "bm25.pkl"))
        sym_index.save(str(save_dir / "symbols.db"))

        console.print(f"\n  [green]✓ Saved to {save_dir}[/green]")

    report = tracker.report()
    console.print(f"\n[dim]Build time: {report.index_build_time_sec:.1f}s[/dim]")
    console.print(f"[dim]Memory: {report.peak_memory_mb:.1f} MB peak[/dim]")

    # Print summary table
    table = Table(title="Repository Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Files parsed", str(len(parsed_files)))
    table.add_row("Symbols", str(len(symbols)))
    table.add_row("Graph nodes", str(stats["total_nodes"]))
    table.add_row("Graph edges", str(stats["total_edges"]))
    table.add_row("Pages", str(page_index.page_count))

    for ntype, count in sorted(stats.get("node_types", {}).items()):
        table.add_row(f"  {ntype}", str(count))
    for etype, count in sorted(stats.get("edge_types", {}).items()):
        table.add_row(f"  edge: {etype}", str(count))

    console.print(table)


# ─── Query Command ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--repo", default=None, help="Repository name (index to use)")
@click.option("--strategy", type=click.Choice(["vector", "vectorless", "hybrid"]),
              default="vectorless", help="Retrieval strategy")
@click.option("--top-k", default=10, help="Number of results to retrieve")
@click.option("--llm/--no-llm", default=False, help="Generate LLM answer")
@click.option("--model", default="nomic-code",
              type=click.Choice(["nomic-code", "bge-m3", "nomic-text", "bge-base", "minilm"]),
              help="HuggingFace embedding model (for vector/hybrid strategies)")
def query(query, repo, strategy, top_k, llm, model):
    """Query the code graph with natural language."""
    # Discover repo index
    if repo is None:
        repos = [d.name for d in INDEX_DIR.iterdir() if d.is_dir()]
        if not repos:
            console.print("[red]No indexed repositories found. Run 'ingest' first.[/red]")
            sys.exit(1)
        repo = repos[0]
        console.print(f"[dim]Using repository: {repo}[/dim]")

    save_dir = INDEX_DIR / repo
    if not save_dir.exists():
        console.print(f"[red]Index not found: {save_dir}[/red]")
        sys.exit(1)

    # Load indexes
    console.print("[dim]Loading indexes...[/dim]")
    graph = CodeGraph()
    graph.load(str(save_dir / "graph.pkl"))

    bm25_index = BM25CodeIndex()
    bm25_index.load(str(save_dir / "bm25.pkl"))

    sym_index = SymbolIndex()
    sym_index.load(str(save_dir / "symbols.db"))

    page_index = PageIndex()
    page_index.build(graph.all_symbols)

    # Build retriever
    if strategy == "vector":
        vec_index = VectorCodeIndex(
            collection_name=repo,
            persist_dir=str(INDEX_DIR / "vector_store"),
            model_key=model,
        )
        retriever = VectorRetriever(vec_index)
    elif strategy == "vectorless":
        retriever = VectorlessRetriever(
            bm25_index=bm25_index,
            symbol_index=sym_index,
            page_index=page_index,
            code_graph=graph,
        )
    else:  # hybrid
        vec_index = VectorCodeIndex(
            collection_name=repo,
            persist_dir=str(INDEX_DIR / "vector_store"),
            model_key=model,
        )
        retriever = HybridRetriever(
            bm25_index=bm25_index,
            vector_index=vec_index,
            symbol_index=sym_index,
            page_index=page_index,
            code_graph=graph,
        )

    # Retrieve
    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[bold]Strategy:[/bold] {retriever.name}\n")

    result = retriever.retrieve(query, top_k=top_k)

    # Display results
    table = Table(title=f"Top {len(result.pages)} Results ({result.retrieval_time_ms:.1f}ms)")
    table.add_column("#", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Symbol", width=30)
    table.add_column("Type", width=10)
    table.add_column("File", width=50)
    table.add_column("Lines", width=10)

    for i, sp in enumerate(result.pages):
        p = sp.page
        table.add_row(
            str(i + 1),
            f"{sp.score:.4f}",
            p.symbol_name,
            p.symbol_type,
            p.file_path,
            f"{p.line_start}-{p.line_end}",
        )

    console.print(table)

    # Metadata
    if result.metadata:
        console.print(f"\n[dim]Metadata: {json.dumps(result.metadata, default=str, indent=2)}[/dim]")

    # LLM answer (optional)
    if llm:
        console.print("\n[bold]Generating LLM answer...[/bold]")
        ctx_builder = ContextBuilder(code_graph=graph)
        context = ctx_builder.build(result.pages, query=query)
        prompt = CONTEXT_TEMPLATE.format(context=context, query=query)

        llm_client = LLMClient()
        response = llm_client.query(prompt, system_prompt=SYSTEM_PROMPT)

        console.print(f"\n[bold green]Answer:[/bold green]\n{response.content}")
        console.print(f"\n[dim]Tokens: {response.total_tokens} | "
                      f"Latency: {response.latency_ms:.0f}ms[/dim]")


# ─── Stats Command ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--repo", default=None, help="Repository name")
def stats(repo):
    """Show statistics for an indexed repository."""
    if repo is None:
        repos = [d.name for d in INDEX_DIR.iterdir() if d.is_dir()]
        if not repos:
            console.print("[red]No indexed repositories.[/red]")
            return
        for r in repos:
            console.print(f"  • {r}")
        return

    save_dir = INDEX_DIR / repo
    graph = CodeGraph()
    graph.load(str(save_dir / "graph.pkl"))

    s = graph.stats()
    console.print(f"\n[bold]{repo}[/bold]")
    console.print(json.dumps(s, indent=2))


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
