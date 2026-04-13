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
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    intersection = retrieved_k & relevant
    return len(intersection) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K — fraction of top-k results that are relevant.
    Precision@K = |relevant ∩ retrieved@k| / k
    """
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    intersection = retrieved_k & relevant
    return len(intersection) / k


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Mean Reciprocal Rank (MRR) — reciprocal of the rank of the first relevant item.
    MRR = 1 / rank_i (where i is first relevant)
    Returns max reciprocal rank for multiple relevant items.
    """
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG) at K.
    Measures both presence and position of relevant items.
    """
    import math
    if not relevant:
        return 0.0
        
    actual_dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            actual_dcg += 1.0 / math.log2(i + 2)
            
    ideal_dcg = 0.0
    for i in range(min(len(relevant), k)):
        ideal_dcg += 1.0 / math.log2(i + 2)
        
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Hit@K — binary: is any relevant item in the top-k results?"""
    retrieved_k = set(retrieved[:k])
    return 1.0 if (retrieved_k & relevant) else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. QA METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exact_match(predicted: str, ground_truth: str) -> float:
    """Exact Match (EM) — 1.0 if answer exactly matches ground truth."""
    pred_norm = " ".join(predicted.lower().split())
    truth_norm = " ".join(ground_truth.lower().split())
    return 1.0 if pred_norm == truth_norm else 0.0


def f1_score(predicted: str, ground_truth: str) -> float:
    """F1 Score — token-level overlap between predicted and ground truth."""
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


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single retrieval method."""
    method_name: str
    num_queries: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_score: float = 0.0
    latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "queries": self.num_queries,
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr_score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "p95": round(self.p95_latency_ms, 2),
        }

def compute_all_metrics(
    retrieved_lists: List[List[str]],
    relevant_sets: List[Set[str]],
    latencies: Optional[List[float]] = None,
    method_name: str = "unknown",
) -> EvaluationResult:
    n = len(retrieved_lists)
    result = EvaluationResult(method_name=method_name, num_queries=n)
    if n == 0:
        return result

    r5s, r10s, mrrs = [], [], []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        r5s.append(recall_at_k(retrieved, relevant, 5))
        r10s.append(recall_at_k(retrieved, relevant, 10))
        mrrs.append(mrr(retrieved, relevant))

    result.recall_at_5 = sum(r5s) / n
    result.recall_at_10 = sum(r10s) / n
    result.mrr_score = sum(mrrs) / n

    if latencies:
        result.latency_ms = sum(latencies) / len(latencies)
        sorted_lats = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_lats))
        result.p95_latency_ms = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

    return result
