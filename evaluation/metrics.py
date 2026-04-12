"""
Evaluation metrics for retrieval and QA performance.

Implements all standard IR and QA metrics used in RAG research:

Retrieval Metrics:
  - Recall@K
  - Precision@K
  - MRR (Mean Reciprocal Rank)
  - nDCG@K (Normalized Discounted Cumulative Gain)
  - Hit@K (Hit Rate)

QA Metrics:
  - Exact Match (EM)
  - F1 Score (token overlap)
  - Hallucination Rate

All metrics follow TREC / CodeSearchNet / RAG benchmark conventions.
"""

import math
import logging
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. RETRIEVAL METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K — fraction of relevant items found in top-k results.

    Recall@K = |relevant ∩ retrieved@k| / |relevant|

    Most important metric in RAG research.
    Standard K values: 1, 5, 10
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K — fraction of top-k results that are relevant.

    Precision@K = |relevant ∩ retrieved@k| / k
    """
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.

    MRR = 1 / rank_of_first_relevant

    Standard metric in code search research (CodeSearchNet, GraphCodeBERT).
    Emphasizes early retrieval quality.
    """
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    nDCG@K — Normalized Discounted Cumulative Gain.

    DCG@K = Σ(i=1..k) relevance_i / log2(i + 1)
    nDCG@K = DCG@K / IDCG@K

    Used when multiple documents have varying relevance levels.
    Falls back to binary relevance if no scores provided.
    """
    def dcg(items: List[str], limit: int) -> float:
        score = 0.0
        for i, item in enumerate(items[:limit]):
            if relevance_scores:
                rel = relevance_scores.get(item, 0.0)
            else:
                rel = 1.0 if item in relevant else 0.0
            score += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return score

    # DCG of actual retrieval
    actual_dcg = dcg(retrieved, k)

    # IDCG: best possible ordering
    if relevance_scores:
        ideal_order = sorted(relevant, key=lambda x: relevance_scores.get(x, 0.0), reverse=True)
    else:
        ideal_order = list(relevant)

    ideal_dcg = dcg(ideal_order, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Hit@K — binary: is any relevant item in the top-k results?

    Hit@K = 1 if |relevant ∩ retrieved@k| > 0, else 0

    Simple but widely used in RAG and recommendation system papers.
    """
    retrieved_k = set(retrieved[:k])
    return 1.0 if (retrieved_k & relevant) else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. QA METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exact_match(predicted: str, ground_truth: str) -> float:
    """
    Exact Match (EM) — 1.0 if answer exactly matches ground truth.

    Used in SQuAD-style QA evaluation.
    Normalizes whitespace and case.
    """
    pred_norm = " ".join(predicted.lower().split())
    truth_norm = " ".join(ground_truth.lower().split())
    return 1.0 if pred_norm == truth_norm else 0.0


def f1_score(predicted: str, ground_truth: str) -> float:
    """
    F1 Score — token-level overlap between predicted and ground truth.

    F1 = 2 * (precision * recall) / (precision + recall)

    Standard metric in open-domain QA and retrieval-augmented QA.
    """
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def hallucination_rate(
    answers: List[str],
    ground_truths: List[str],
    hallucination_flags: Optional[List[bool]] = None,
) -> float:
    """
    Hallucination Rate — fraction of answers containing unsupported claims.

    Hallucination Rate = |hallucinated answers| / |total answers|

    If hallucination_flags not provided, uses F1 < 0.1 as a proxy
    for hallucination detection.
    """
    if hallucination_flags:
        return sum(hallucination_flags) / len(hallucination_flags)

    # Proxy: very low F1 suggests fabricated content
    hallucinated = 0
    for pred, truth in zip(answers, ground_truths):
        if f1_score(pred, truth) < 0.1:
            hallucinated += 1

    return hallucinated / max(len(answers), 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. BATCH EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single retrieval method."""
    method_name: str
    num_queries: int = 0

    # Retrieval metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr_score: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0

    # QA metrics
    exact_match_score: float = 0.0
    f1_avg: float = 0.0
    hallucination_rate_score: float = 0.0

    # Efficiency metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    memory_mb: float = 0.0

    # Per-query scores (for statistical testing)
    per_query_recall_5: List[float] = field(default_factory=list)
    per_query_mrr: List[float] = field(default_factory=list)
    per_query_f1: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dict (excluding per-query lists)."""
        return {
            "method": self.method_name,
            "queries": self.num_queries,
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "precision@5": round(self.precision_at_5, 4),
            "precision@10": round(self.precision_at_10, 4),
            "mrr": round(self.mrr_score, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "hit@1": round(self.hit_at_1, 4),
            "hit@5": round(self.hit_at_5, 4),
            "hit@10": round(self.hit_at_10, 4),
            "em": round(self.exact_match_score, 4),
            "f1": round(self.f1_avg, 4),
            "hallucination_rate": round(self.hallucination_rate_score, 4),
            "latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "tokens": round(self.avg_tokens, 1),
            "memory_mb": round(self.memory_mb, 1),
        }


def compute_all_metrics(
    retrieved_lists: List[List[str]],       # List of retrieved page_ids per query
    relevant_sets: List[Set[str]],          # List of relevant page_ids per query
    predicted_answers: Optional[List[str]] = None,
    ground_truth_answers: Optional[List[str]] = None,
    latencies: Optional[List[float]] = None,
    method_name: str = "unknown",
) -> EvaluationResult:
    """
    Compute all evaluation metrics over a batch of queries.

    This is the main entry point for evaluation — pass in per-query
    retrieved results and ground truth, get back all standard metrics.
    """
    n = len(retrieved_lists)
    result = EvaluationResult(method_name=method_name, num_queries=n)

    if n == 0:
        return result

    # Retrieval metrics — average over queries
    r1s, r5s, r10s = [], [], []
    p5s, p10s = [], []
    mrrs = []
    n5s, n10s = [], []
    h1s, h5s, h10s = [], [], []

    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        r1s.append(recall_at_k(retrieved, relevant, 1))
        r5s.append(recall_at_k(retrieved, relevant, 5))
        r10s.append(recall_at_k(retrieved, relevant, 10))
        p5s.append(precision_at_k(retrieved, relevant, 5))
        p10s.append(precision_at_k(retrieved, relevant, 10))
        mrrs.append(mrr(retrieved, relevant))
        n5s.append(ndcg_at_k(retrieved, relevant, 5))
        n10s.append(ndcg_at_k(retrieved, relevant, 10))
        h1s.append(hit_at_k(retrieved, relevant, 1))
        h5s.append(hit_at_k(retrieved, relevant, 5))
        h10s.append(hit_at_k(retrieved, relevant, 10))

    result.recall_at_1 = sum(r1s) / n
    result.recall_at_5 = sum(r5s) / n
    result.recall_at_10 = sum(r10s) / n
    result.precision_at_5 = sum(p5s) / n
    result.precision_at_10 = sum(p10s) / n
    result.mrr_score = sum(mrrs) / n
    result.ndcg_at_5 = sum(n5s) / n
    result.ndcg_at_10 = sum(n10s) / n
    result.hit_at_1 = sum(h1s) / n
    result.hit_at_5 = sum(h5s) / n
    result.hit_at_10 = sum(h10s) / n

    # Per-query scores for statistical testing
    result.per_query_recall_5 = r5s
    result.per_query_mrr = mrrs

    # QA metrics
    if predicted_answers and ground_truth_answers:
        ems = [exact_match(p, g) for p, g in zip(predicted_answers, ground_truth_answers)]
        f1s = [f1_score(p, g) for p, g in zip(predicted_answers, ground_truth_answers)]
        result.exact_match_score = sum(ems) / n
        result.f1_avg = sum(f1s) / n
        result.per_query_f1 = f1s
        result.hallucination_rate_score = hallucination_rate(
            predicted_answers, ground_truth_answers
        )

    # Latency metrics
    if latencies:
        result.avg_latency_ms = sum(latencies) / len(latencies)
        sorted_lats = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_lats))
        result.p95_latency_ms = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

    return result
