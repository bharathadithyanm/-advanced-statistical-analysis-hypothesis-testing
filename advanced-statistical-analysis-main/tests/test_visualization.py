"""
Unit tests for visualization module.
"""

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import (
    plot_distribution_overview,
    plot_normality_diagnostics,
    plot_hypothesis_comparison,
    plot_anova_distributions,
    plot_two_way_anova_interaction,
    plot_tukey_hsd_results,
    plot_anova_diagnostics,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    return rng.integers(1, 15, size=200)


def test_plot_distribution_overview(tmp_path, sample_data):
    out = os.path.join(tmp_path, "dist.png")
    fig = plot_distribution_overview(sample_data, save_path=out)
    assert os.path.exists(out)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_normality_diagnostics(tmp_path, sample_data):
    out = os.path.join(tmp_path, "norm.png")
    fig = plot_normality_diagnostics(sample_data, save_path=out)
    assert os.path.exists(out)
    plt.close(fig)


def test_plot_hypothesis_comparison(tmp_path):
    out = os.path.join(tmp_path, "hypo.png")
    g1 = [2, 3, 4, 5, 6]
    g2 = [3, 4, 5, 6, 7]
    fig = plot_hypothesis_comparison(g1, g2, save_path=out)
    assert os.path.exists(out)
    plt.close(fig)


def test_plot_tukey_hsd_results(tmp_path):
    out = os.path.join(tmp_path, "tukey.png")
    df = pd.DataFrame({
        "group1": ["A", "A", "B"],
        "group2": ["B", "C", "C"],
        "mean_diff": [0.2, 1.3, 1.1],
        "p_adjusted": [0.65, 0.001, 0.003],
        "ci_lower": [-0.3, 0.8, 0.6],
        "ci_upper": [0.7, 1.8, 1.6],
        "reject_null": [False, True, True]
    })
    fig = plot_tukey_hsd_results(df, save_path=out)
    assert os.path.exists(out)
    plt.close(fig)
