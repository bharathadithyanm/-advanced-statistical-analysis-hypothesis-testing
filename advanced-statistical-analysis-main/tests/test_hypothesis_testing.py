"""
Unit tests for hypothesis_testing module.
"""

import pytest
import numpy as np
from src.hypothesis_testing import (
    check_variance_homogeneity,
    welch_t_test,
    mann_whitney_u_test,
    compare_parametric_vs_nonparametric,
)


@pytest.fixture
def sample_pair():
    rng = np.random.default_rng(42)
    # Group 1: mean=5.0, std=1.5
    g1 = rng.normal(loc=5.0, scale=1.5, size=500)
    # Group 2: mean=4.0, std=2.5 (different variance and mean)
    g2 = rng.normal(loc=4.0, scale=2.5, size=600)
    return g1, g2


def test_homogeneity_of_variance(sample_pair):
    g1, g2 = sample_pair
    res = check_variance_homogeneity(g1, g2, center="median")
    assert res["equal_variance_justified"] is False
    assert res["p_value"] < 0.05
    assert "Welch" in res["recommended_t_test"]


def test_welch_t_test(sample_pair):
    g1, g2 = sample_pair
    res = welch_t_test(g1, g2, group1_name="Treated", group2_name="Control")
    assert np.isclose(res["mean_diff"], 1.0, atol=0.3)
    assert res["ci_lower"] < res["mean_diff"] < res["ci_upper"]
    assert res["p_value"] < 0.001
    assert res["reject_null"] is True
    assert res["cohens_d_pooled"] > 0.3
    assert res["df"] > 500


def test_welch_t_test_identical():
    g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    g2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = welch_t_test(g1, g2)
    assert np.isclose(res["mean_diff"], 0.0)
    assert res["p_value"] == 1.0
    assert res["reject_null"] is False


def test_mann_whitney_u_test(sample_pair):
    g1, g2 = sample_pair
    res = mann_whitney_u_test(g1, g2, group1_name="Treated", group2_name="Control")
    assert res["p_value"] < 0.001
    assert res["reject_null"] is True
    assert np.abs(res["rank_biserial_r"]) > 0.1
    assert res["z_score"] > 3.0


def test_compare_parametric_vs_nonparametric(sample_pair):
    g1, g2 = sample_pair
    t_res = welch_t_test(g1, g2)
    m_res = mann_whitney_u_test(g1, g2)
    comp = compare_parametric_vs_nonparametric(t_res, m_res)
    assert comp["concordance"] is True
    assert "Methodological Synthesis" in comp["explanation"]
