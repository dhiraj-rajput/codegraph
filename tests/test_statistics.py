"""
Tests for statistical significance tests.
"""

import pytest
from evaluation.statistics import paired_t_test, bootstrap_test, compare_methods


class TestPairedTTest:
    def test_identical_scores(self):
        """Identical scores → not significant."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = paired_t_test(scores, scores, method_a="A", method_b="B", metric="recall@5")
        assert not result.significant
        assert result.p_value > 0.05

    def test_clearly_different(self):
        """Clearly different scores → significant."""
        a = [0.9, 0.85, 0.95, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93, 0.90]
        b = [0.3, 0.35, 0.28, 0.32, 0.30, 0.33, 0.29, 0.31, 0.34, 0.27]
        result = paired_t_test(a, b, method_a="A", method_b="B", metric="recall@5")
        assert result.significant
        assert result.p_value < 0.05
        assert result.effect_size > 1.0  # Large effect

    def test_output_structure(self):
        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        result = paired_t_test(a, b, method_a="X", method_b="Y", metric="mrr")
        assert result.test_name == "Paired t-test"
        assert result.method_a == "X"
        assert result.method_b == "Y"
        assert result.metric == "mrr"
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)


class TestBootstrapTest:
    def test_identical_scores(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_test(scores, scores, method_a="A", method_b="B",
                               metric="recall@5", n_bootstrap=1000)
        assert not result.significant

    def test_different_scores(self):
        a = [0.9, 0.85, 0.95, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93, 0.90]
        b = [0.3, 0.35, 0.28, 0.32, 0.30, 0.33, 0.29, 0.31, 0.34, 0.27]
        result = bootstrap_test(a, b, method_a="A", method_b="B",
                               metric="recall@5", n_bootstrap=1000)
        assert result.significant


class TestCompareMethods:
    def test_pairwise_comparison(self):
        results = {
            "BM25": [0.5, 0.6, 0.7, 0.8, 0.9],
            "Vector": [0.6, 0.7, 0.8, 0.9, 0.95],
            "Hybrid": [0.65, 0.75, 0.85, 0.92, 0.97],
        }
        tests = compare_methods(results, metric="recall@5")
        # 3 methods → 3 pairwise tests (3 choose 2)
        assert len(tests) == 3
