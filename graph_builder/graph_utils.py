"""
Graph traversal utilities for retrieval-time expansion.

Provides multiple traversal strategies beyond basic BFS:
- Random Walk with Restart (for Personalized PageRank retrieval)
- Depth-limited DFS
- Weighted shortest path
"""

import random
import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


def random_walk_with_restart(
    graph: nx.DiGraph,
    seed_node: str,
    num_steps: int = 100,
    restart_prob: float = 0.15,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """
    Random walk with restart from a seed node.

    At each step, either restart at the seed (with probability `restart_prob`)
    or move to a random neighbor. Returns visit frequency per node.

    Useful for finding nodes "related" to a seed in a more nuanced way
    than simple BFS, especially in dense graphs.
    """
    rng = random.Random(rng_seed)
    visit_count: Dict[str, int] = defaultdict(int)

    current = seed_node
    if current not in graph:
        return {}

    for _ in range(num_steps):
        visit_count[current] += 1

        if rng.random() < restart_prob:
            current = seed_node
        else:
            neighbors = list(graph.successors(current)) + list(graph.predecessors(current))
            if neighbors:
                current = rng.choice(neighbors)
            else:
                current = seed_node

    # Normalize to frequencies
    total = sum(visit_count.values())
    return {node: count / total for node, count in visit_count.items()}


def depth_limited_dfs(
    graph: nx.DiGraph,
    seed_node: str,
    depth: int = 3,
    max_nodes: int = 50,
) -> List[str]:
    """
    Depth-limited DFS from a seed node.

    Returns node IDs reachable within `depth` hops via DFS.
    Uses both forward and backward edges (undirected traversal).
    """
    visited: Set[str] = set()
    result: List[str] = []

    def _dfs(node: str, d: int):
        if d < 0 or node in visited or len(result) >= max_nodes:
            return
        visited.add(node)
        result.append(node)

        neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
        for n in neighbors:
            _dfs(n, d - 1)

    if seed_node in graph:
        _dfs(seed_node, depth)

    return result


def subgraph_around_nodes(
    graph: nx.DiGraph,
    seed_nodes: List[str],
    radius: int = 2,
) -> nx.DiGraph:
    """
    Extract the subgraph within `radius` hops of any seed node.

    Useful for visualization and context-aware retrieval.
    """
    all_nodes: Set[str] = set()

    for seed in seed_nodes:
        if seed not in graph:
            continue
        # Forward reachability
        try:
            for node, dist in nx.single_source_shortest_path_length(graph, seed, cutoff=radius).items():
                all_nodes.add(node)
        except nx.NetworkXError:
            pass
        # Backward reachability
        try:
            rev = graph.reverse(copy=False)
            for node, dist in nx.single_source_shortest_path_length(rev, seed, cutoff=radius).items():
                all_nodes.add(node)
        except nx.NetworkXError:
            pass

    return graph.subgraph(all_nodes).copy()


def compute_personalized_pagerank(
    graph: nx.DiGraph,
    seed_nodes: List[str],
    alpha: float = 0.85,
) -> Dict[str, float]:
    """
    Personalized PageRank with seed nodes as personalization set.

    Nodes structurally "close" to seeds get higher scores.
    Useful as a graph-proximity signal in retrieval ranking.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {}

    personalization = {}
    valid_seeds = [s for s in seed_nodes if s in graph]
    if not valid_seeds:
        return {}

    for node in graph.nodes:
        personalization[node] = 1.0 if node in valid_seeds else 0.0

    # Normalize
    total = sum(personalization.values())
    if total > 0:
        personalization = {k: v / total for k, v in personalization.items()}

    try:
        return nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Personalized PageRank failed to converge")
        return {node: 1.0 / n for node in graph.nodes}
