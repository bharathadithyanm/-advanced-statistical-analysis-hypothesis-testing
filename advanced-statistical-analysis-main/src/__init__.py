"""
Advanced Statistical Analysis and Hypothesis Testing Package.
"""

from .normality_tests import (
    compute_distribution_summary,
    shapiro_wilk_test,
    repeated_shapiro_wilk,
    kolmogorov_smirnov_test,
    run_comprehensive_normality_suite,
)

from .hypothesis_testing import (
    check_variance_homogeneity,
    welch_t_test,
    mann_whitney_u_test,
    compare_parametric_vs_nonparametric,
)

from .anova_analysis import (
    one_way_anova,
    two_way_anova,
    tukey_hsd_post_hoc,
)

from .visualization import (
    plot_distribution_overview,
    plot_normality_diagnostics,
    plot_hypothesis_comparison,
    plot_anova_distributions,
    plot_two_way_anova_interaction,
    plot_tukey_hsd_results,
    plot_anova_diagnostics,
)

__all__ = [
    "compute_distribution_summary",
    "shapiro_wilk_test",
    "repeated_shapiro_wilk",
    "kolmogorov_smirnov_test",
    "run_comprehensive_normality_suite",
    "check_variance_homogeneity",
    "welch_t_test",
    "mann_whitney_u_test",
    "compare_parametric_vs_nonparametric",
    "one_way_anova",
    "two_way_anova",
    "tukey_hsd_post_hoc",
    "plot_distribution_overview",
    "plot_normality_diagnostics",
    "plot_hypothesis_comparison",
    "plot_anova_distributions",
    "plot_two_way_anova_interaction",
    "plot_tukey_hsd_results",
    "plot_anova_diagnostics",
]
