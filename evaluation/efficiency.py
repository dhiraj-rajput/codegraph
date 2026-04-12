"""
Efficiency tracking — latency, memory usage, index build time, token cost.

These metrics are critical for demonstrating the practical advantages
of vectorless retrieval over embedding-based approaches.
"""

import time
import logging
import tracemalloc
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyReport:
    """Complete efficiency report for a retrieval method."""
    method_name: str

    # Retrieval latency
    retrieval_latencies_ms: List[float] = field(default_factory=list)
    avg_retrieval_latency_ms: float = 0.0
    p95_retrieval_latency_ms: float = 0.0
    median_retrieval_latency_ms: float = 0.0

    # Index build time
    index_build_time_sec: float = 0.0

    # Memory usage
    index_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Token cost
    total_context_tokens: int = 0
    total_output_tokens: int = 0
    avg_context_tokens: float = 0.0

    def compute_stats(self):
        """Compute derived statistics from raw measurements."""
        lats = self.retrieval_latencies_ms
        if lats:
            self.avg_retrieval_latency_ms = sum(lats) / len(lats)
            sorted_lats = sorted(lats)
            self.median_retrieval_latency_ms = sorted_lats[len(sorted_lats) // 2]
            p95_idx = int(0.95 * len(sorted_lats))
            self.p95_retrieval_latency_ms = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "avg_latency_ms": round(self.avg_retrieval_latency_ms, 2),
            "p95_latency_ms": round(self.p95_retrieval_latency_ms, 2),
            "median_latency_ms": round(self.median_retrieval_latency_ms, 2),
            "index_build_time_sec": round(self.index_build_time_sec, 2),
            "index_memory_mb": round(self.index_memory_mb, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "avg_context_tokens": round(self.avg_context_tokens, 1),
            "total_tokens": self.total_context_tokens + self.total_output_tokens,
        }


class EfficiencyTracker:
    """
    Tracks system efficiency metrics across experiments.

    Usage:
        tracker = EfficiencyTracker("vectorless")
        with tracker.measure_build():
            build_index(...)
        with tracker.measure_retrieval():
            retriever.retrieve(query)
        report = tracker.report()
    """

    def __init__(self, method_name: str):
        self.method_name = method_name
        self._latencies: List[float] = []
        self._build_time: float = 0.0
        self._context_tokens: List[int] = []
        self._output_tokens: List[int] = []
        self._peak_memory: float = 0.0
        self._index_memory: float = 0.0

    @contextmanager
    def measure_build(self):
        """Context manager to measure index build time and memory."""
        try:
            tracemalloc.start()
        except Exception:
            pass

        start = time.perf_counter()
        yield
        self._build_time = time.perf_counter() - start

        try:
            current, peak = tracemalloc.get_traced_memory()
            self._index_memory = current / 1024 / 1024  # MB
            self._peak_memory = peak / 1024 / 1024
            tracemalloc.stop()
        except Exception:
            pass

    @contextmanager
    def measure_retrieval(self):
        """Context manager to measure a single retrieval latency."""
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms
        self._latencies.append(elapsed)

    def record_retrieval_time(self, ms: float):
        """Record a retrieval latency directly."""
        self._latencies.append(ms)

    def record_tokens(self, context_tokens: int, output_tokens: int = 0):
        """Record token usage for a single query."""
        self._context_tokens.append(context_tokens)
        self._output_tokens.append(output_tokens)

    def measure_memory(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def report(self) -> EfficiencyReport:
        """Generate the final efficiency report."""
        report = EfficiencyReport(
            method_name=self.method_name,
            retrieval_latencies_ms=self._latencies,
            index_build_time_sec=self._build_time,
            index_memory_mb=self._index_memory,
            peak_memory_mb=self._peak_memory,
            total_context_tokens=sum(self._context_tokens),
            total_output_tokens=sum(self._output_tokens),
            avg_context_tokens=(
                sum(self._context_tokens) / len(self._context_tokens)
                if self._context_tokens else 0.0
            ),
        )
        report.compute_stats()
        return report
