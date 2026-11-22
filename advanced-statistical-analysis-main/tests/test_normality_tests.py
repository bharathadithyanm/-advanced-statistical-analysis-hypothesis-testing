"""
Unit tests for normality_tests module.
"""

import pytest
import numpy as np
import scipy.stats as stats
from src.normality_tests import (
    compute_distribution_summary,
    shapiro_wilk_test,
    repeated_shapiro_wilk,
    kolmogorov_smirnov_test,
    run_comprehensive_normality_suite,
)


@pytest.fixture
def normal_sample():
    rng = np.random.default_rng(42)
    return rng.normal(loc=10.0, scale=2.5, size=1000)


@pytest.fixture
def skewed_sample():
    rng = np.random.default_rng(42)
    return rng.exponential(scale=3.0, size=1000)


def test_compute_distribution_summary(normal_sample):
    summary = compute_distribution_summary(normal_sample, variable_name="test_norm")
    assert summary["n"] == 1000
    assert np.isclose(summary["mean"], 10.0, atol=0.25)
    assert np.isclose(summary["std"], 2.5, atol=0.25)
    assert np.abs(summary["skewness"]) < 0.2
    assert summary["min"] < summary["median"] < summary["max"]


def test_compute_distribution_summary_empty():
    with pytest.raises(ValueError):
        compute_distribution_summary([])


def test_shapiro_wilk_normal(normal_sample):
    res = shapiro_wilk_test(normal_sample, sample_size=1000, random_state=42)
    assert res["statistic"] > 0.95
    assert res["p_value"] > 0.05
    assert res["reject_null"] is False


def test_shapiro_wilk_skewed(skewed_sample):
    res = shapiro_wilk_test(skewed_sample, sample_size=1000, random_state=42)
    assert res["statistic"] < 0.95
    assert res["p_value"] < 0.05
    assert res["reject_null"] is True


def test_shapiro_wilk_subsampling():
    rng = np.random.default_rng(123)
    large_sample = rng.normal(5, 1, size=8000)
    res = shapiro_wilk_test(large_sample, sample_size=3000, random_state=42)
    assert res["was_sampled"] is True
    assert res["effective_n"] == 3000
    assert res["total_n"] == 8000


def test_repeated_shapiro_wilk(skewed_sample):
    res = repeated_shapiro_wilk(skewed_sample, n_iterations=10, sample_size=500, random_state=42)
    assert res["iterations"] == 10
    assert res["rejection_rate"] == 1.0


def test_kolmogorov_smirnov_normal(normal_sample):
    res = kolmogorov_smirnov_test(normal_sample)
    assert res["statistic"] < 0.05
    assert res["p_value"] > 0.05
    assert res["reject_null"] is False


def test_kolmogorov_smirnov_skewed(skewed_sample):
    res = kolmogorov_smirnov_test(skewed_sample)
    assert res["p_value"] < 0.001
    assert res["reject_null"] is True


def test_run_comprehensive_normality_suite(normal_sample):
    res = run_comprehensive_normality_suite(normal_sample, variable_name="sim_norm")
    assert "summary" in res
    assert "shapiro_wilk" in res
    assert "repeated_shapiro_wilk" in res
    assert "kolmogorov_smirnov" in res
