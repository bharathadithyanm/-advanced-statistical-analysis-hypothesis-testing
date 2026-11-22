"""
ANOVA analysis module for advanced statistical analysis.
Implements One-Way ANOVA, Two-Way Factorial ANOVA with interaction effects,
model diagnostics (residual normality, homoscedasticity), effect sizes (eta-squared, partial eta-squared),
and Tukey HSD post-hoc pairwise multiple comparisons.
"""

from typing import Dict, Any, Tuple, Union, Optional, List
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def one_way_anova(
    df: pd.DataFrame,
    group_col: str = "A1Cresult",
    value_col: str = "time_in_hospital",
    target_groups: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform One-Way ANOVA comparing continuous values across categorical levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    group_col : str
        Name of categorical factor column.
    value_col : str
        Name of numeric response column.
    target_groups : list, optional
        Specific group levels to retain (e.g., ['Norm', '>7', '>8']).
    alpha : float
        Significance threshold.
        
    Returns
    -------
    dict
        F-statistic, p-value, group summaries, effect sizes (eta^2, omega^2),
        and assumption test results (Levene test, residual normality).
    """
    data = df[[group_col, value_col]].dropna().copy()
    if target_groups is not None:
        data = data[data[group_col].isin(target_groups)].copy()
        
    # Group statistics
    group_stats = []
    group_arrays = []
    unique_groups = sorted(data[group_col].unique().tolist())
    
    for g in unique_groups:
        vals = data.loc[data[group_col] == g, value_col].values
        group_arrays.append(vals)
        group_stats.append({
            "group": g,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25))
        })
        
    group_stats_df = pd.DataFrame(group_stats)
    
    # One-Way ANOVA using scipy and statsmodels OLS for complete sum of squares
    f_stat, p_val = stats.f_oneway(*group_arrays)
    f_stat = float(f_stat)
    p_val = float(p_val)
    
    # OLS formulation for ANOVA table and residuals
    formula = f"{value_col} ~ C({group_col})"
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    ss_between = float(anova_table.loc[f"C({group_col})", "sum_sq"])
    ss_within = float(anova_table.loc["Residual", "sum_sq"])
    ss_total = ss_between + ss_within
    
    df_between = int(anova_table.loc[f"C({group_col})", "df"])
    df_within = int(anova_table.loc["Residual", "df"])
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # Effect sizes: eta-squared (proportion of variance explained)
    eta_squared = ss_between / ss_total
    # Omega-squared (unbiased population effect size)
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    
    # Model assumptions
    # 1. Homogeneity of variance (Levene test)
    levene_stat, levene_p = stats.levene(*group_arrays, center="median")
    
    # 2. Residual normality
    residuals = model.resid.values
    ks_res = stats.kstest((residuals - np.mean(residuals)) / np.std(residuals, ddof=1), "norm")
    
    # Sample 5000 residuals for Shapiro-Wilk if N > 5000
    if len(residuals) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(residuals), size=5000, replace=False)
        sw_stat, sw_p = stats.shapiro(residuals[idx])
    else:
        sw_stat, sw_p = stats.shapiro(residuals)
        
    reject_null = bool(p_val < alpha)
    interpretation = (
        f"Statistically significant difference in {value_col} across groups ({', '.join(unique_groups)}), "
        f"F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_val:.4e} < {alpha}. "
        f"Effect size is small (eta^2 = {eta_squared:.4f}, omega^2 = {omega_squared:.4f}). "
        f"Post-hoc pairwise testing is indicated."
        if reject_null
        else f"No statistically significant difference in {value_col} across groups "
             f"(F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_val:.4f} >= {alpha})."
    )
    
    return {
        "test_name": f"One-Way ANOVA ({value_col} by {group_col})",
        "group_col": group_col,
        "value_col": value_col,
        "groups": unique_groups,
        "group_stats": group_stats_df,
        "f_statistic": f_stat,
        "p_value": p_val,
        "df_between": df_between,
        "df_within": df_within,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "eta_squared": float(eta_squared),
        "omega_squared": float(omega_squared),
        "alpha": alpha,
        "null_hypothesis": f"Population means of {value_col} are equal across all levels of {group_col}.",
        "alternative_hypothesis": f"At least one group population mean of {value_col} differs from the others.",
        "reject_null": reject_null,
        "levene_statistic": float(levene_stat),
        "levene_p_value": float(levene_p),
        "residuals_ks_stat": float(ks_res.statistic),
        "residuals_ks_p_value": float(ks_res.pvalue),
        "residuals_sw_stat": float(sw_stat),
        "residuals_sw_p_value": float(sw_p),
        "residuals": residuals,
        "fitted_values": model.fittedvalues.values,
        "model": model,
        "interpretation": interpretation
    }


def two_way_anova(
    df: pd.DataFrame,
    factor1_col: str = "a1c_tested",
    factor2_col: str = "admission_type",
    value_col: str = "time_in_hospital",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform Two-Way Factorial ANOVA evaluating Main Effects and Interaction Effect.
    
    Uses Type II sum of squares (appropriate for unbalanced designs with no hierarchy).
    Calculates partial eta-squared for each effect.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    factor1_col, factor2_col : str
        Categorical factor column names.
    value_col : str
        Numeric dependent variable.
    alpha : float
        Significance threshold.
        
    Returns
    -------
    dict
        ANOVA table, partial eta-squared effect sizes, cell means, model diagnostics.
    """
    data = df[[factor1_col, factor2_col, value_col]].dropna().copy()
    
    # OLS Model with interaction term
    formula = f"{value_col} ~ C({factor1_col}) + C({factor2_col}) + C({factor1_col}):C({factor2_col})"
    model = ols(formula, data=data).fit()
    
    # Type II Sum of Squares
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    ss_error = float(anova_table.loc["Residual", "sum_sq"])
    df_error = int(anova_table.loc["Residual", "df"])
    
    # Extract terms and partial eta-squared: SS_effect / (SS_effect + SS_error)
    effects = []
    term1 = f"C({factor1_col})"
    term2 = f"C({factor2_col})"
    term_inter = f"C({factor1_col}):C({factor2_col})"
    
    for term, label in [
        (term1, f"Main Effect: {factor1_col}"),
        (term2, f"Main Effect: {factor2_col}"),
        (term_inter, f"Interaction: {factor1_col} x {factor2_col}")
    ]:
        ss = float(anova_table.loc[term, "sum_sq"])
        df_term = int(anova_table.loc[term, "df"])
        f_val = float(anova_table.loc[term, "F"])
        p_val = float(anova_table.loc[term, "PR(>F)"])
        p_eta2 = ss / (ss + ss_error)
        
        effects.append({
            "term": label,
            "sum_sq": ss,
            "df": df_term,
            "f_statistic": f_val,
            "p_value": p_val,
            "partial_eta_squared": float(p_eta2),
            "reject_null": bool(p_val < alpha)
        })
        
    effects_df = pd.DataFrame(effects)
    
    # Cell means and counts
    cell_stats = data.groupby([factor1_col, factor2_col])[value_col].agg(
        n="count",
        mean="mean",
        std="std",
        median="median"
    ).reset_index()
    
    # Residual diagnostic tests
    residuals = model.resid.values
    ks_res = stats.kstest((residuals - np.mean(residuals)) / np.std(residuals, ddof=1), "norm")
    
    # Levene test across all cell combinations
    cells = [group[value_col].values for _, group in data.groupby([factor1_col, factor2_col])]
    levene_stat, levene_p = stats.levene(*cells, center="median")
    
    return {
        "test_name": f"Two-Way ANOVA ({value_col} by {factor1_col} & {factor2_col})",
        "factor1_col": factor1_col,
        "factor2_col": factor2_col,
        "value_col": value_col,
        "anova_table": anova_table,
        "effects_summary": effects_df,
        "cell_stats": cell_stats,
        "residuals": residuals,
        "fitted_values": model.fittedvalues.values,
        "model": model,
        "levene_statistic": float(levene_stat),
        "levene_p_value": float(levene_p),
        "residuals_ks_stat": float(ks_res.statistic),
        "residuals_ks_p_value": float(ks_res.pvalue),
        "alpha": alpha
    }


def tukey_hsd_post_hoc(
    df: pd.DataFrame,
    group_col: str = "A1Cresult",
    value_col: str = "time_in_hospital",
    target_groups: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform Tukey's Honestly Significant Difference (HSD) post-hoc pairwise multiple comparisons test.
    
    Controls Family-Wise Error Rate (FWER) across all pairwise combinations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    group_col : str
        Categorical grouping column.
    value_col : str
        Numeric response variable.
    target_groups : list, optional
        Specific groups to evaluate.
    alpha : float
        Family-wise significance level (default 0.05).
        
    Returns
    -------
    dict
        Pairwise comparisons, mean differences, adjusted p-values, 95% CIs, and statistical decisions.
    """
    data = df[[group_col, value_col]].dropna().copy()
    if target_groups is not None:
        data = data[data[group_col].isin(target_groups)].copy()
        
    tukey = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=alpha)
    
    # Convert Tukey HSD results to clean DataFrame
    results_table = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )
    # Ensure standard column naming
    rename_dict = {
        "group1": "group1",
        "group2": "group2",
        "meandiff": "mean_diff",
        "p-adj": "p_adjusted",
        "lower": "ci_lower",
        "upper": "ci_upper",
        "reject": "reject_null"
    }
    results_df = results_table.rename(columns=rename_dict)
    for col in ["mean_diff", "p_adjusted", "ci_lower", "ci_upper"]:
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(float)
    if "reject_null" in results_df.columns:
        results_df["reject_null"] = results_df["reject_null"].astype(bool)
        
    return {
        "test_name": "Tukey Honestly Significant Difference (HSD) Post-Hoc Test",
        "group_col": group_col,
        "value_col": value_col,
        "alpha": alpha,
        "tukey_object": tukey,
        "comparisons_table": results_df,
        "summary_text": str(tukey.summary())
    }
