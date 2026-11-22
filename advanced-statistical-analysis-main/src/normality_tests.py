"""
Normality testing module for advanced statistical analysis.
Implements Shapiro-Wilk, Kolmogorov-Smirnov, repeated subsampling, and distribution moments.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_distribution_summary(
    data: Union[np.ndarray, pd.Series, list],
    variable_name: str = "variable"
) -> Dict[str, Any]:
    """
    Compute descriptive statistics, moments, and shape characteristics of the data.
    
    Parameters
    ----------
    data : array-like
        Numeric array or Series of observations.
    variable_name : str
        Human-readable name of the variable.
        
    Returns
    -------
    dict
        Dictionary containing sample size, mean, std, median, IQR, min, max,
        skewness, excess kurtosis, and standard error of moments.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    
    if n == 0:
        raise ValueError("Cannot compute distribution summary on empty array.")
        
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    median_val = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr_val = q75 - q25
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    
    # Skewness and excess kurtosis (Fisher definition where normal = 0)
    skew_val = float(stats.skew(arr, bias=False)) if n > 2 else float(stats.skew(arr))
    kurt_val = float(stats.kurtosis(arr, bias=False)) if n > 3 else float(stats.kurtosis(arr))
    
    # Standard errors of skewness and kurtosis under asymptotic normality
    se_skew = np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 3 else np.nan
    se_kurt = 2.0 * se_skew * np.sqrt((n**2 - 1) / ((n - 3) * (n + 5))) if n > 3 else np.nan
    
    return {
        "variable_name": variable_name,
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "q25": q25,
        "q75": q75,
        "iqr": iqr_val,
        "min": min_val,
        "max": max_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "se_skewness": float(se_skew) if not np.isnan(se_skew) else None,
        "se_kurtosis": float(se_kurt) if not np.isnan(se_kurt) else None
    }


def shapiro_wilk_test(
    data: Union[np.ndarray, pd.Series, list],
    sample_size: Optional[int] = 5000,
    random_state: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the Shapiro-Wilk test for normality.
    
    For large datasets (N > 5000), applies a representative random sampling strategy
    to avoid algorithmic constraints and excessive hyper-sensitivity to trivial deviations.
    
    Parameters
    ----------
    data : array-like
        Numeric array of observations.
    sample_size : int, optional
        Sample size to draw if len(data) > sample_size. Default 5000.
        If None or if len(data) <= sample_size, runs on all data.
    random_state : int
        RNG seed for reproducible subsampling.
    alpha : float
        Significance threshold (default 0.05).
        
    Returns
    -------
    dict
        Structured results including statistic, p-value, hypotheses, and decision.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    total_n = len(arr)
    
    if total_n < 3:
        raise ValueError("Shapiro-Wilk test requires at least 3 non-missing observations.")
        
    was_sampled = False
    effective_sample_size = total_n
    
    if sample_size is not None and total_n > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(total_n, size=sample_size, replace=False)
        test_arr = arr[sample_idx]
        effective_sample_size = sample_size
        was_sampled = True
    else:
        test_arr = arr
        
    stat, p_val = stats.shapiro(test_arr)
    stat = float(stat)
    p_val = float(p_val)
    reject_null = bool(p_val < alpha)
    
    interpretation = (
        f"Statistically significant deviation from normality (p = {p_val:.4e} < {alpha}). "
        f"The null hypothesis of normality is rejected."
        if reject_null
        else f"Fail to reject normality (p = {p_val:.4f} >= {alpha}). "
             f"Data does not show statistically significant departure from normality."
    )
    
    return {
        "test_name": "Shapiro-Wilk Test",
        "total_n": total_n,
        "effective_n": effective_sample_size,
        "was_sampled": was_sampled,
        "statistic": stat,
        "p_value": p_val,
        "alpha": alpha,
        "null_hypothesis": "The population distribution is normally distributed.",
        "alternative_hypothesis": "The population distribution is not normally distributed.",
        "reject_null": reject_null,
        "interpretation": interpretation
    }


def repeated_shapiro_wilk(
    data: Union[np.ndarray, pd.Series, list],
    n_iterations: int = 100,
    sample_size: int = 2000,
    random_state: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Monte Carlo repeated subsampling Shapiro-Wilk test to evaluate stability across random draws.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    total_n = len(arr)
    
    rng = np.random.default_rng(random_state)
    stats_list = []
    pvals_list = []
    
    effective_size = min(sample_size, total_n)
    for _ in range(n_iterations):
        sample_idx = rng.choice(total_n, size=effective_size, replace=False)
        s, p = stats.shapiro(arr[sample_idx])
        stats_list.append(s)
        pvals_list.append(p)
        
    stats_arr = np.array(stats_list)
    pvals_arr = np.array(pvals_list)
    rejection_rate = float(np.mean(pvals_arr < alpha))
    
    return {
        "test_name": "Repeated Subsampling Shapiro-Wilk",
        "iterations": n_iterations,
        "sample_size_per_iter": effective_size,
        "mean_statistic": float(np.mean(stats_arr)),
        "std_statistic": float(np.std(stats_arr)),
        "mean_p_value": float(np.mean(pvals_arr)),
        "median_p_value": float(np.median(pvals_arr)),
        "rejection_rate": rejection_rate,
        "alpha": alpha
    }


def kolmogorov_smirnov_test(
    data: Union[np.ndarray, pd.Series, list],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the Kolmogorov-Smirnov test against a fitted continuous normal distribution.
    
    Standardizes the data using sample mean and sample standard deviation, then
    compares empirical CDF against the theoretical standard normal CDF.
    
    Parameters
    ----------
    data : array-like
        Numeric array of observations.
    alpha : float
        Significance threshold (default 0.05).
        
    Returns
    -------
    dict
        Structured results including KS statistic, p-value, hypotheses, and decision.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    
    if n < 5:
        raise ValueError("Kolmogorov-Smirnov test requires at least 5 observations.")
        
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    
    # Standardized values (Z-scores)
    z_scores = (arr - mean_val) / std_val
    
    ks_stat, p_val = stats.kstest(z_scores, "norm")
    ks_stat = float(ks_stat)
    p_val = float(p_val)
    reject_null = bool(p_val < alpha)
    
    interpretation = (
        f"Statistically significant deviation from normality (KS stat = {ks_stat:.4f}, p = {p_val:.4e} < {alpha}). "
        f"The null hypothesis of normality is rejected."
        if reject_null
        else f"Fail to reject normality (KS stat = {ks_stat:.4f}, p = {p_val:.4f} >= {alpha}). "
             f"Data does not significantly deviate from a normal distribution."
    )
    
    return {
        "test_name": "Kolmogorov-Smirnov Test (vs Fitted Normal)",
        "total_n": n,
        "statistic": ks_stat,
        "p_value": p_val,
        "alpha": alpha,
        "fitted_mean": float(mean_val),
        "fitted_std": float(std_val),
        "null_hypothesis": "The data follows a normal distribution with specified parameters.",
        "alternative_hypothesis": "The data does not follow the specified normal distribution.",
        "reject_null": reject_null,
        "interpretation": interpretation
    }


def run_comprehensive_normality_suite(
    data: Union[np.ndarray, pd.Series, list],
    variable_name: str = "time_in_hospital",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Execute complete normality evaluation suite on a variable:
    moments, Shapiro-Wilk (sampled), repeated Shapiro-Wilk, and Kolmogorov-Smirnov.
    """
    summary = compute_distribution_summary(data, variable_name=variable_name)
    sw_result = shapiro_wilk_test(data, sample_size=5000, alpha=alpha)
    rep_sw_result = repeated_shapiro_wilk(data, n_iterations=50, sample_size=2000, alpha=alpha)
    ks_result = kolmogorov_smirnov_test(data, alpha=alpha)
    
    return {
        "variable_name": variable_name,
        "summary": summary,
        "shapiro_wilk": sw_result,
        "repeated_shapiro_wilk": rep_sw_result,
        "kolmogorov_smirnov": ks_result
    }
