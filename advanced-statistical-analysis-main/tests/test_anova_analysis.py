"""
Unit tests for anova_analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from src.anova_analysis import (
    one_way_anova,
    two_way_anova,
    tukey_hsd_post_hoc,
)


@pytest.fixture
def anova_dataset():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "group": np.repeat(["A", "B", "C"], n),
        "factor2": np.tile(np.repeat(["Low", "High"], n // 2), 3),
        "value": np.concatenate([
            rng.normal(loc=4.0, scale=1.0, size=n),  # A
            rng.normal(loc=4.2, scale=1.0, size=n),  # B
            rng.normal(loc=5.5, scale=1.0, size=n),  # C (elevated)
        ])
    })
    return df


def test_one_way_anova(anova_dataset):
    res = one_way_anova(anova_dataset, group_col="group", value_col="value")
    assert res["f_statistic"] > 50.0
    assert res["p_value"] < 0.0001
    assert res["reject_null"] is True
    assert res["eta_squared"] > 0.1
    assert res["df_between"] == 2
    assert res["df_within"] == 597
    assert len(res["group_stats"]) == 3


def test_two_way_anova(anova_dataset):
    res = two_way_anova(anova_dataset, factor1_col="group", factor2_col="factor2", value_col="value")
    effects = res["effects_summary"]
    assert len(effects) == 3
    # Main effect of group should be highly significant
    grp_effect = effects.loc[effects["term"].str.contains("group") & ~effects["term"].str.contains("x")].iloc[0]
    assert grp_effect["p_value"] < 0.001
    assert bool(grp_effect["reject_null"]) is True
    assert len(res["cell_stats"]) == 6


def test_tukey_hsd_post_hoc(anova_dataset):
    res = tukey_hsd_post_hoc(anova_dataset, group_col="group", value_col="value")
    table = res["comparisons_table"]
    assert len(table) == 3  # (3 choose 2) comparisons
    # Comparison A vs C must be rejected
    ac_row = table[((table["group1"] == "A") & (table["group2"] == "C")) | ((table["group1"] == "C") & (table["group2"] == "A"))].iloc[0]
    assert bool(ac_row["reject_null"]) is True
    assert ac_row["p_adjusted"] < 0.001
