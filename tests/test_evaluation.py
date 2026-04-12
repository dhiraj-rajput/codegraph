"""
Tests for the evaluation metrics module.

Verifies all IR and QA metrics against known expected values
to ensure correctness before running experiments.
"""

import pytest
from evaluation.metrics import (
    recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_at_k,
    exact_match, f1_score, hallucination_rate,
    compute_all_metrics,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 5) == 1.0
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)

    def test_no_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_recall_at_1(self):
        retrieved = ["a", "b", "c"]
        relevant = {"b", "c"}
        assert recall_at_k(retrieved, relevant, 1) == 0.0
        assert recall_at_k(retrieved, relevant, 2) == 0.5

    def test_recall_at_5(self):
        retrieved = ["x", "a", "y", "b", "z"]
        relevant = {"a", "b", "c"}
        # At k=5: found a, b (2 out of 3)
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_precision(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_no_precision(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_precision_at_5(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(3 / 5)


class TestMRR:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mrr(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        assert mrr(retrieved, relevant) == 0.5

    def test_not_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert mrr(retrieved, relevant) == 0.0

    def test_example_from_spec(self):
        # MRR = (0.5 + 1 + 0.2) / 3
        queries = [
            (["x", "a"], {"a"}),            # rank 2 -> 1/2
            (["a", "b"], {"a"}),            # rank 1 -> 1
            (["x", "y", "z", "w", "a"], {"a"}),  # rank 5 -> 1/5
        ]
        scores = [mrr(r, rel) for r, rel in queries]
        avg_mrr = sum(scores) / len(scores)
        assert avg_mrr == pytest.approx((0.5 + 1.0 + 0.2) / 3, abs=0.01)


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_partial_ranking(self):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        # DCG = 0/log2(2) + 1/log2(3) = 0.631
        # IDCG = 1/log2(2) = 1.0
        result = ndcg_at_k(retrieved, relevant, 3)
        assert 0.0 < result < 1.0


class TestHitAtK:
    def test_hit(self):
        assert hit_at_k(["a", "b"], {"a"}, 2) == 1.0

    def test_miss(self):
        assert hit_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_hit_at_1(self):
        assert hit_at_k(["b", "a"], {"a"}, 1) == 0.0
        assert hit_at_k(["a", "b"], {"a"}, 1) == 1.0


class TestExactMatch:
    def test_exact(self):
        assert exact_match("validateJWT", "validateJWT") == 1.0

    def test_case_insensitive(self):
        assert exact_match("ValidateJWT", "validatejwt") == 1.0

    def test_no_match(self):
        assert exact_match("foo", "bar") == 0.0

    def test_whitespace_normalization(self):
        assert exact_match("hello  world", "hello world") == 1.0


class TestF1Score:
    def test_perfect_match(self):
        assert f1_score("the cat sat", "the cat sat") == 1.0

    def test_partial_match(self):
        result = f1_score("the cat", "the cat sat on mat")
        assert 0.0 < result < 1.0

    def test_no_overlap(self):
        assert f1_score("foo bar", "baz qux") == 0.0

    def test_empty(self):
        assert f1_score("", "hello") == 0.0


class TestHallucinationRate:
    def test_no_hallucination(self):
        rate = hallucination_rate(
            ["the answer is A", "the answer is B"],
            ["the answer is A", "the answer is B"],
        )
        assert rate == 0.0

    def test_with_flags(self):
        rate = hallucination_rate(
            ["a", "b", "c"],
            ["a", "b", "c"],
            hallucination_flags=[True, False, False],
        )
        assert rate == pytest.approx(1 / 3)


class TestComputeAllMetrics:
    def test_basic(self):
        retrieved_lists = [
            ["a", "b", "c"],
            ["x", "a", "y"],
        ]
        relevant_sets = [
            {"a", "b"},
            {"a"},
        ]
        result = compute_all_metrics(
            retrieved_lists, relevant_sets, method_name="test"
        )
        assert result.num_queries == 2
        assert result.recall_at_1 > 0
        assert result.mrr_score > 0
        assert result.method_name == "test"

    def test_with_qa_metrics(self):
        result = compute_all_metrics(
            retrieved_lists=[["a"]],
            relevant_sets=[{"a"}],
            predicted_answers=["the cat sat"],
            ground_truth_answers=["the cat sat on the mat"],
            method_name="qa_test",
        )
        assert result.f1_avg > 0
        assert result.exact_match_score == 0.0  # Not perfect match
