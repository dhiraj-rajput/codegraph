"""
Statistical significance testing for evaluation results.

Required for publishable research — proves that performance differences
between methods are not due to random chance.

Implements:
- Paired t-test (parametric)
- Bootstrap test (non-parametric)
"""

import logging
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatTestResult:
    """Result of a statistical significance test."""
    test_name: str
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    statistic: float           # t-statistic or bootstrap difference
    p_value: float
    significant: bool          # True if p < alpha
    alpha: float = 0.05
    effect_size: float = 0.0   # Cohen's d

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.test_name}: {self.method_a} vs {self.method_b} on {self.metric}\n"
            f"  Mean A={self.mean_a:.4f}, Mean B={self.mean_b:.4f}\n"
            f"  p={self.p_value:.6f} ({sig} at α={self.alpha})\n"
            f"  Effect size (Cohen's d) = {self.effect_size:.4f}"
        )


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    method_a: str = "A",
    method_b: str = "B",
    metric: str = "unknown",
    alpha: float = 0.05,
) -> StatTestResult:
    """
    Paired t-test for comparing two methods on the same queries.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have equal length"
    a = np.array(scores_a)
    b = np.array(scores_b)

    differences = a - b
    std_diff = np.std(differences, ddof=1)

    # Handle identical scores perfectly
    if std_diff == 0:
        p_value = 1.0 if np.all(a == b) else 0.0
        t_stat = 0.0
    else:
        t_stat, p_value = stats.ttest_rel(a, b)

    # Cohen's d for paired samples
    d_mean = np.mean(differences)
    cohens_d = d_mean / std_diff if std_diff > 0 else 0.0

    # Handle potential nan p_value if still present
    if np.isnan(p_value):
        p_value = 1.0

    return StatTestResult(
        test_name="Paired t-test",
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        statistic=float(t_stat),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        alpha=alpha,
        effect_size=float(cohens_d),
    )


def bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    method_a: str = "A",
    method_b: str = "B",
    metric: str = "unknown",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> StatTestResult:
    """
    Randomized Paired Test (a robust bootstrap variant).

    Tests the null hypothesis that the mean difference is zero by 
    randomly flipping the signs of the observed differences.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have equal length"

    np.random.seed(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)

    differences = a - b
    observed_mean_diff = np.mean(differences)

    if np.all(differences == 0):
        return StatTestResult(
            test_name="Bootstrap test", method_a=method_a, method_b=method_b,
            metric=metric, mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
            statistic=0.0, p_value=1.0, significant=False, alpha=alpha, effect_size=0.0
        )

    # Generate bootstrap samples under H0 by flipping signs
    count_more_extreme = 0
    for _ in range(n_bootstrap):
        # Randomly flip signs of differences (H0: mean diff is 0)
        signs = np.random.choice([-1, 1], size=n)
        boot_diff = np.mean(differences * signs)

        if abs(boot_diff) >= abs(observed_mean_diff):
            count_more_extreme += 1

    p_value = count_more_extreme / n_bootstrap

    # Effect size
    d_std = np.std(differences, ddof=1)
    cohens_d = observed_mean_diff / d_std if d_std > 0 else 0.0

    return StatTestResult(
        test_name="Bootstrap test",
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        statistic=float(observed_mean_diff),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        alpha=alpha,
        effect_size=float(cohens_d),
    )


def compare_methods(
    results: dict,          # method_name -> per_query_scores
    metric: str = "recall@5",
    alpha: float = 0.05,
) -> List[StatTestResult]:
    """
    Compare all pairs of methods with paired t-test.

    Returns a list of StatTestResult for every method pair.
    """
    methods = list(results.keys())
    all_tests = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            result = paired_t_test(
                scores_a=results[methods[i]],
                scores_b=results[methods[j]],
                method_a=methods[i],
                method_b=methods[j],
                metric=metric,
                alpha=alpha,
            )
            all_tests.append(result)

    return all_tests
