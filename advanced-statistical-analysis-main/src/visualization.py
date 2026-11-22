"""
Visualization module for advanced statistical analysis and hypothesis testing.
Produces publication-quality plots adhering to scientific reporting standards.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use headless Agg backend
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Configure styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
sns.set_theme(style="whitegrid", palette="deep")


def plot_distribution_overview(
    data: Union[np.ndarray, pd.Series, list],
    variable_name: str = "time_in_hospital",
    title_prefix: str = "Hospital Stay Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate a 4-panel overview: Histogram, KDE with Normal Fit, Q-Q Plot, Box Plot, and Violin Plot.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} ({variable_name.replace('_', ' ').title()})", fontsize=16, fontweight="bold", y=0.98)
    
    # 1. Histogram with KDE and fitted Normal
    ax1 = axes[0, 0]
    sns.histplot(arr, kde=True, stat="density", discrete=True, color="#2b5c8f", alpha=0.6, ax=ax1, label="Empirical Density")
    # Fitted normal curve
    mu, sigma = np.mean(arr), np.std(arr, ddof=1)
    x_grid = np.linspace(min(arr) - 1, max(arr) + 1, 300)
    normal_curve = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
    ax1.plot(x_grid, normal_curve, color="#d95f02", linestyle="--", linewidth=2.2, label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})")
    ax1.set_title("A: Histogram & Empirical KDE vs Normal Fit", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"{variable_name.replace('_', ' ').title()} (Days)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    
    # 2. Q-Q Plot
    ax2 = axes[0, 1]
    (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist="norm")
    ax2.plot(osm, osr, "o", color="#2b5c8f", alpha=0.4, markersize=3.5, label="Empirical Quantiles")
    ax2.plot(osm, slope * np.array(osm) + intercept, color="#d95f02", linestyle="-", linewidth=2, label=f"Reference Line (R²={r**2:.3f})")
    ax2.set_title("B: Normal Quantile-Quantile (Q-Q) Plot", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles (Standard Normal)", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    
    # 3. Box Plot
    ax3 = axes[1, 0]
    sns.boxplot(x=arr, ax=ax3, color="#7570b3", flierprops={"marker": "d", "markerfacecolor": "#e7298a", "markersize": 4})
    ax3.set_title("C: Box Plot (Median & IQR Distribution)", fontsize=12, fontweight="bold")
    ax3.set_xlabel(f"{variable_name.replace('_', ' ').title()} (Days)", fontsize=11)
    # Annotate summary moments
    med = np.median(arr)
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    ax3.text(0.05, 0.85, f"Median: {med:.1f}\nIQR: {q75 - q25:.1f} ({q25:.1f}-{q75:.1f})\nMean: {mu:.2f} ± {sigma:.2f}",
             transform=ax3.transAxes, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#bbbbbb", alpha=0.9))
    
    # 4. Violin Plot
    ax4 = axes[1, 1]
    sns.violinplot(x=arr, ax=ax4, color="#1b9e77", inner="quartile", cut=0)
    ax4.set_title("D: Violin Plot (Kernel Density & Quartiles)", fontsize=12, fontweight="bold")
    ax4.set_xlabel(f"{variable_name.replace('_', ' ').title()} (Days)", fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_normality_diagnostics(
    data: Union[np.ndarray, pd.Series, list],
    variable_name: str = "time_in_hospital",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Focused 2-panel normality diagnostics: Density & Q-Q Plot with statistical annotations.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    mu, sigma = np.mean(arr), np.std(arr, ddof=1)
    skew_val = stats.skew(arr)
    kurt_val = stats.kurtosis(arr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Normality Diagnostics for {variable_name.replace('_', ' ').title()}", fontsize=15, fontweight="bold")
    
    # Density plot
    sns.histplot(arr, kde=True, stat="density", discrete=True, color="#1f77b4", alpha=0.5, ax=ax1, label="Observed KDE")
    x_pts = np.linspace(min(arr) - 1, max(arr) + 1, 300)
    ax1.plot(x_pts, stats.norm.pdf(x_pts, mu, sigma), color="#d62728", linestyle="--", linewidth=2.2, label="Theoretical Gaussian")
    ax1.set_title("Empirical Distribution vs Theoretical Normal", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Hospital Stay (Days)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.text(0.55, 0.55, f"Sample N: {len(arr):,}\nMean: {mu:.2f}, SD: {sigma:.2f}\nSkewness: {skew_val:.3f}\nExcess Kurtosis: {kurt_val:.3f}",
             transform=ax1.transAxes, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9), fontsize=10)
    
    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist="norm")
    ax2.plot(osm, osr, "o", color="#1f77b4", alpha=0.3, markersize=3, label="Data Quantiles")
    ax2.plot(osm, slope * np.array(osm) + intercept, color="#d62728", linewidth=2, label="Normal Line")
    ax2.set_title("Normal Q-Q Plot", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Standard Normal Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles (Days)", fontsize=11)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_hypothesis_comparison(
    group1: Union[np.ndarray, pd.Series, list],
    group2: Union[np.ndarray, pd.Series, list],
    name1: str = "Tested",
    name2: str = "Not Tested",
    value_name: str = "Hospital Stay (Days)",
    cohens_d: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    3-panel comparative hypothesis testing figure:
    Panel 1: Side-by-side Box Plot with jittered points preview.
    Panel 2: Split / Dual Violin Plot.
    Panel 3: Mean Comparison with 95% Confidence Intervals.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    # Prepare DataFrame
    plot_df = pd.DataFrame({
        "Group": [name1] * len(g1) + [name2] * len(g2),
        "Value": np.concatenate([g1, g2])
    })
    
    palette = {name1: "#2b5c8f", name2: "#d95f02"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Comparison of Hospital Stay by Inpatient HbA1c Testing Status", fontsize=15, fontweight="bold")
    
    # Panel 1: Box Plot
    ax1 = axes[0]
    sns.boxplot(data=plot_df, x="Group", y="Value", hue="Group", palette=palette, ax=ax1, width=0.45,
                legend=False, flierprops={"marker": ".", "alpha": 0.2, "markersize": 3})
    ax1.set_title("A: Box Plot Distributions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("HbA1c Testing Category", fontsize=11)
    ax1.set_ylabel(value_name, fontsize=11)
    # Annotate medians
    med1, med2 = np.median(g1), np.median(g2)
    ax1.text(0, med1 + 0.3, f"Median: {med1:.1f}", ha="center", fontweight="bold", color="#1c3b5e")
    ax1.text(1, med2 + 0.3, f"Median: {med2:.1f}", ha="center", fontweight="bold", color="#8c3d01")
    
    # Panel 2: Violin Plot
    ax2 = axes[1]
    sns.violinplot(data=plot_df, x="Group", y="Value", hue="Group", palette=palette, ax=ax2, inner="quartile", cut=0, legend=False)
    ax2.set_title("B: Violin Density Profiles", fontsize=12, fontweight="bold")
    ax2.set_xlabel("HbA1c Testing Category", fontsize=11)
    ax2.set_ylabel(value_name, fontsize=11)
    
    # Panel 3: Means & 95% CIs
    ax3 = axes[2]
    m1, m2 = np.mean(g1), np.mean(g2)
    se1 = np.std(g1, ddof=1) / np.sqrt(len(g1))
    se2 = np.std(g2, ddof=1) / np.sqrt(len(g2))
    ci1 = 1.96 * se1
    ci2 = 1.96 * se2
    
    x_pos = [0, 1]
    means = [m1, m2]
    cis = [ci1, ci2]
    colors = [palette[name1], palette[name2]]
    
    for x, m, ci, c in zip(x_pos, means, cis, colors):
        ax3.errorbar(x, m, yerr=ci, fmt="o", color=c, ecolor=c, elinewidth=2.5, capsize=6, markersize=9)
        ax3.text(x, m + 0.08, f"Mean: {m:.2f}\n(95% CI: [{m-ci:.2f}, {m+ci:.2f}])", ha="center", fontsize=9.5, fontweight="bold")
        
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{name1}\n(n={len(g1):,})", f"{name2}\n(n={len(g2):,})"], fontsize=10)
    ax3.set_title("C: Mean Stay & 95% Confidence Intervals", fontsize=12, fontweight="bold")
    ax3.set_ylabel(value_name, fontsize=11)
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(min(means) - 0.5, max(means) + 0.6)
    
    if cohens_d is not None:
        ax3.text(0.5, 0.15, f"Effect Size: Cohen's d = {cohens_d:.3f} (Small)\nMean Diff = +{m1-m2:.2f} days",
                 transform=ax3.transAxes, ha="center", bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#888888", alpha=0.9))
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_anova_distributions(
    df: pd.DataFrame,
    group_col: str = "A1Cresult",
    value_col: str = "time_in_hospital",
    target_groups: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize One-Way ANOVA group distributions across HbA1c categories with Boxplot and Violin plot.
    """
    data = df[[group_col, value_col]].dropna().copy()
    if target_groups is not None:
        data = data[data[group_col].isin(target_groups)].copy()
        
    order = target_groups if target_groups is not None else sorted(data[group_col].unique().tolist())
    palette = sns.color_palette("deep", n_colors=len(order))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"One-Way ANOVA: Distribution of Hospital Stay Across {group_col} Levels", fontsize=15, fontweight="bold")
    
    # 1. Boxplot with group mean markers
    sns.boxplot(data=data, x=group_col, y=value_col, hue=group_col, order=order, palette=palette, ax=ax1, width=0.5,
                legend=False, showmeans=True, meanprops={"marker": "s", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 7})
    ax1.set_title("Group Distributions (Boxplot & Mean Markers)", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"{group_col} Level", fontsize=11)
    ax1.set_ylabel("Hospital Stay (Days)", fontsize=11)
    
    # Annotate group means and sample sizes
    for i, grp in enumerate(order):
        sub = data.loc[data[group_col] == grp, value_col]
        m, n = sub.mean(), len(sub)
        ax1.text(i, m + 0.4, f"Mean: {m:.2f}\n(n={n:,})", ha="center", fontsize=9, fontweight="bold")
        
    # 2. Violin Plot
    sns.violinplot(data=data, x=group_col, y=value_col, hue=group_col, order=order, palette=palette, ax=ax2, inner="quartile", cut=0, legend=False)
    ax2.set_title("Density Distributions (Violin Plot)", fontsize=12, fontweight="bold")
    ax2.set_xlabel(f"{group_col} Level", fontsize=11)
    ax2.set_ylabel("Hospital Stay (Days)", fontsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_two_way_anova_interaction(
    df: pd.DataFrame,
    factor1_col: str = "a1c_tested",
    factor2_col: str = "admission_type",
    value_col: str = "time_in_hospital",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot interaction effects and marginal means for Two-Way ANOVA.
    """
    data = df[[factor1_col, factor2_col, value_col]].dropna().copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Two-Way ANOVA: Main & Interaction Effects on {value_col.replace('_', ' ').title()}", fontsize=15, fontweight="bold")
    
    # 1. Interaction Plot: Mean hospital stay by factor1 stratified by factor2
    sns.pointplot(
        data=data, x=factor1_col, y=value_col, hue=factor2_col,
        dodge=0.2, errorbar=("ci", 95), capsize=0.1, markers=["o", "s", "^"], linestyles=["-", "--", "-."],
        palette="tab10", ax=ax1
    )
    ax1.set_title(f"A: Interaction Plot ({factor1_col} x {factor2_col})", fontsize=12, fontweight="bold")
    ax1.set_xlabel(factor1_col.replace("_", " ").title(), fontsize=11)
    ax1.set_ylabel("Mean Hospital Stay (Days) [95% CI]", fontsize=11)
    ax1.legend(title=factor2_col.replace("_", " ").title(), loc="upper right")
    
    # 2. Reverse interaction plot: Factor 2 on x-axis stratified by Factor 1
    sns.pointplot(
        data=data, x=factor2_col, y=value_col, hue=factor1_col,
        dodge=0.2, errorbar=("ci", 95), capsize=0.1, markers=["o", "D"], linestyles=["-", ":"],
        palette=["#2b5c8f", "#d95f02"], ax=ax2
    )
    ax2.set_title(f"B: Stratified Marginal Means by {factor2_col}", fontsize=12, fontweight="bold")
    ax2.set_xlabel(factor2_col.replace("_", " ").title(), fontsize=11)
    ax2.set_ylabel("Mean Hospital Stay (Days) [95% CI]", fontsize=11)
    ax2.legend(title=factor1_col.replace("_", " ").title(), loc="upper right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_tukey_hsd_results(
    tukey_table: pd.DataFrame,
    title: str = "Tukey HSD Post-Hoc Pairwise Differences (95% CI)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate a forest plot of Tukey HSD pairwise mean differences with 95% confidence intervals.
    """
    df_plot = tukey_table.copy()
    df_plot["comparison"] = df_plot["group1"].astype(str) + " vs " + df_plot["group2"].astype(str)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(df_plot))
    
    for i, row in df_plot.iterrows():
        diff = row["mean_diff"]
        lower = row["ci_lower"]
        upper = row["ci_upper"]
        reject = row["reject_null"]
        p_adj = row["p_adjusted"]
        
        color = "#d95f02" if reject else "#2b5c8f"
        marker = "s" if reject else "o"
        label_text = f"p-adj = {p_adj:.4f} ({'Reject H0' if reject else 'Fail to Reject'})"
        
        ax.errorbar(
            diff, i, xerr=[[diff - lower], [upper - diff]], fmt=marker, color=color,
            ecolor=color, elinewidth=2.2, capsize=6, markersize=8
        )
        ax.text(upper + 0.02, i, f" {diff:+.3f} [{lower:+.3f}, {upper:+.3f}] | {label_text}",
                va="center", fontsize=9.5, fontweight="bold" if reject else "normal", color=color)
        
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["comparison"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Difference in Hospital Stay (Days) [Group 2 - Group 1]", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    
    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d95f02", markersize=8, label="Significant (Reject H0, p < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2b5c8f", markersize=8, label="Non-Significant (p >= 0.05)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    
    # Set x limits with margin for text
    x_min = df_plot["ci_lower"].min() - 0.1
    x_max = df_plot["ci_upper"].max() + 0.6
    ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_anova_diagnostics(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    model_name: str = "One-Way ANOVA",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate residual diagnostics for ANOVA:
    Panel 1: Residuals vs Fitted (homogeneity of variance).
    Panel 2: Residual Q-Q Plot (residual normality).
    """
    res = np.asarray(residuals, dtype=float)
    fit = np.asarray(fitted_values, dtype=float)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{model_name} Model Diagnostics: Residual Assumptions", fontsize=15, fontweight="bold")
    
    # Residuals vs Fitted
    ax1.scatter(fit, res, alpha=0.2, color="#2b5c8f", s=15)
    ax1.axhline(0, color="#d95f02", linestyle="--", linewidth=1.8)
    ax1.set_title("A: Residuals vs Fitted Values (Homoscedasticity)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fitted Values (Days)", fontsize=11)
    ax1.set_ylabel("Residuals (Observed - Fitted)", fontsize=11)
    
    # Q-Q plot of residuals
    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
    ax2.plot(osm, osr, "o", color="#2b5c8f", alpha=0.3, markersize=3)
    ax2.plot(osm, slope * np.array(osm) + intercept, color="#d95f02", linewidth=2, label=f"R² = {r**2:.3f}")
    ax2.set_title("B: Normal Q-Q Plot of Residuals", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Standardized Residuals", fontsize=11)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
