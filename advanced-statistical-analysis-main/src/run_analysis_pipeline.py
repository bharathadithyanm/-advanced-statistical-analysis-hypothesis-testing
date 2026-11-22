"""
Complete statistical analysis pipeline script.
Executes Tasks 1 through 4, validates assumptions, and generates all publication-grade figures.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normality_tests import (
    compute_distribution_summary,
    shapiro_wilk_test,
    repeated_shapiro_wilk,
    kolmogorov_smirnov_test,
    run_comprehensive_normality_suite,
)
from src.hypothesis_testing import (
    check_variance_homogeneity,
    welch_t_test,
    mann_whitney_u_test,
    compare_parametric_vs_nonparametric,
)
from src.anova_analysis import (
    one_way_anova,
    two_way_anova,
    tukey_hsd_post_hoc,
)
from src.visualization import (
    plot_distribution_overview,
    plot_normality_diagnostics,
    plot_hypothesis_comparison,
    plot_anova_distributions,
    plot_two_way_anova_interaction,
    plot_tukey_hsd_results,
    plot_anova_diagnostics,
)


def main():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_dir, "data", "diabetic_data.csv")
    viz_dir = os.path.join(project_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f"Loaded raw records: {len(raw_df):,}, columns: {len(raw_df.columns)}")
    
    # 1. Data Cleaning & Validation
    # Exclude invalid/unknown gender records (3 records)
    clean_df = raw_df[raw_df["gender"] != "Unknown/Invalid"].copy()
    print(f"Records after gender validation: {len(clean_df):,}")
    
    # Verify target columns
    required_cols = ["A1Cresult", "time_in_hospital", "admission_type_id", "readmitted", "age", "gender", "num_lab_procedures"]
    for c in required_cols:
        assert c in clean_df.columns, f"Missing column: {c}"
        
    clean_df["time_in_hospital"] = clean_df["time_in_hospital"].astype(int)
    
    # Define A1C testing status: Tested vs Not Tested
    tested_mask = clean_df["A1Cresult"].isin(["Norm", ">7", ">8"])
    clean_df["a1c_tested"] = np.where(tested_mask, "Tested", "Not Tested")
    clean_df["a1c_category"] = clean_df["A1Cresult"].fillna("None")
    
    # Map admission type to top 3 clinical categories
    adm_map = {1: "Emergency", 2: "Urgent", 3: "Elective"}
    clean_df["admission_type"] = clean_df["admission_type_id"].map(adm_map)
    
    # ==========================================
    # TASK 1: Normality Testing
    # ==========================================
    print("\n" + "="*50)
    print("TASK 1: NORMALITY TESTING")
    print("="*50)
    
    hosp_stay = clean_df["time_in_hospital"].values
    norm_suite = run_comprehensive_normality_suite(hosp_stay, variable_name="time_in_hospital", alpha=0.05)
    
    print("Distribution Summary:")
    for k, v in norm_suite["summary"].items():
        print(f"  {k}: {v}")
        
    print("\nShapiro-Wilk Test (Representative Sample N=5,000):")
    print(f"  Statistic W = {norm_suite['shapiro_wilk']['statistic']:.4f}")
    print(f"  p-value = {norm_suite['shapiro_wilk']['p_value']:.4e}")
    print(f"  Reject Normality? {norm_suite['shapiro_wilk']['reject_null']}")
    
    print("\nRepeated Subsampling Shapiro-Wilk (50 iterations x 2,000 observations):")
    print(f"  Mean W = {norm_suite['repeated_shapiro_wilk']['mean_statistic']:.4f} ± {norm_suite['repeated_shapiro_wilk']['std_statistic']:.4f}")
    print(f"  Rejection Rate (alpha=0.05) = {norm_suite['repeated_shapiro_wilk']['rejection_rate']*100:.1f}%")
    
    print("\nKolmogorov-Smirnov Test (Full Dataset N=101,763 vs Fitted Gaussian):")
    print(f"  Statistic D = {norm_suite['kolmogorov_smirnov']['statistic']:.4f}")
    print(f"  p-value = {norm_suite['kolmogorov_smirnov']['p_value']:.4e}")
    print(f"  Reject Normality? {norm_suite['kolmogorov_smirnov']['reject_null']}")
    
    # Plots for Task 1
    p1 = os.path.join(viz_dir, "01_distribution_overview.png")
    plot_distribution_overview(hosp_stay, variable_name="time_in_hospital", save_path=p1)
    print(f"Saved: {p1}")
    
    p2 = os.path.join(viz_dir, "02_normality_diagnostics.png")
    plot_normality_diagnostics(hosp_stay, variable_name="time_in_hospital", save_path=p2)
    print(f"Saved: {p2}")
    
    # ==========================================
    # TASK 2: Hypothesis Testing
    # ==========================================
    print("\n" + "="*50)
    print("TASK 2: HYPOTHESIS TESTING (TESTED VS NOT TESTED)")
    print("="*50)
    
    g_tested = clean_df.loc[clean_df["a1c_tested"] == "Tested", "time_in_hospital"].values
    g_not_tested = clean_df.loc[clean_df["a1c_tested"] == "Not Tested", "time_in_hospital"].values
    
    # Levene's Test
    levene_res = check_variance_homogeneity(g_tested, g_not_tested, center="median", alpha=0.05)
    print(f"Levene's Test: stat={levene_res['statistic']:.4f}, p={levene_res['p_value']:.4e}, equal_var_justified={levene_res['equal_variance_justified']}")
    
    # Welch's t-test
    welch_res = welch_t_test(g_tested, g_not_tested, group1_name="Tested", group2_name="Not Tested", alpha=0.05)
    print("\nWelch's Independent t-test:")
    print(f"  Tested Mean: {welch_res['mean1']:.4f} ± {welch_res['std1']:.4f} (n={welch_res['n1']:,})")
    print(f"  Not Tested Mean: {welch_res['mean2']:.4f} ± {welch_res['std2']:.4f} (n={welch_res['n2']:,})")
    print(f"  Mean Difference: +{welch_res['mean_diff']:.4f} days (SE={welch_res['se_diff']:.4f})")
    print(f"  t({welch_res['df']:.1f}) = {welch_res['statistic']:.4f}, p = {welch_res['p_value']:.4e}")
    print(f"  95% CI: [{welch_res['ci_lower']:.4f}, {welch_res['ci_upper']:.4f}] days")
    print(f"  Cohen's d (pooled) = {welch_res['cohens_d_pooled']:.4f}, Hedges' g = {welch_res['hedges_g']:.4f}")
    
    # Mann-Whitney U Test
    mwu_res = mann_whitney_u_test(g_tested, g_not_tested, group1_name="Tested", group2_name="Not Tested", alpha=0.05)
    print("\nMann-Whitney U Test:")
    print(f"  Tested Median: {mwu_res['median1']:.1f} (IQR {mwu_res['iqr1']:.1f})")
    print(f"  Not Tested Median: {mwu_res['median2']:.1f} (IQR {mwu_res['iqr2']:.1f})")
    print(f"  U Statistic = {mwu_res['statistic']:.1f}, Z = {mwu_res['z_score']:.4f}, p = {mwu_res['p_value']:.4e}")
    print(f"  Rank-Biserial r = {mwu_res['rank_biserial_r']:.4f}, r effect size = {mwu_res['r_effect_size']:.4f}")
    
    # Comparison synthesis
    comparison = compare_parametric_vs_nonparametric(welch_res, mwu_res)
    print("\nMethodological Comparison:\n" + comparison["explanation"])
    
    # Plots for Task 2
    p3 = os.path.join(viz_dir, "03_hypothesis_testing_comparison.png")
    plot_hypothesis_comparison(g_tested, g_not_tested, name1="Tested", name2="Not Tested",
                               cohens_d=welch_res["cohens_d_pooled"], save_path=p3)
    print(f"Saved: {p3}")
    
    # ==========================================
    # TASK 3: ANOVA & TUKEY HSD
    # ==========================================
    print("\n" + "="*50)
    print("TASK 3: ANOVA & TUKEY HSD")
    print("="*50)
    
    # 3A. One-Way ANOVA across 'Norm', '>7', '>8'
    anova_1way = one_way_anova(clean_df, group_col="A1Cresult", value_col="time_in_hospital",
                               target_groups=["Norm", ">7", ">8"], alpha=0.05)
    print("One-Way ANOVA (Tested Patients by A1C Result):")
    print(f"  Groups: {anova_1way['groups']}")
    print(anova_1way["group_stats"].to_string(index=False))
    print(f"  F({anova_1way['df_between']}, {anova_1way['df_within']}) = {anova_1way['f_statistic']:.4f}, p = {anova_1way['p_value']:.4e}")
    print(f"  SS Between = {anova_1way['ss_between']:.2f}, SS Within = {anova_1way['ss_within']:.2f}")
    print(f"  eta-squared = {anova_1way['eta_squared']:.4f}, omega-squared = {anova_1way['omega_squared']:.4f}")
    print(f"  Levene test across groups: stat={anova_1way['levene_statistic']:.4f}, p={anova_1way['levene_p_value']:.4e}")
    print(f"  Residuals KS test: stat={anova_1way['residuals_ks_stat']:.4f}, p={anova_1way['residuals_ks_p_value']:.4e}")
    
    # 3B. Two-Way ANOVA: a1c_tested x admission_type on time_in_hospital
    top3_df = clean_df.dropna(subset=["admission_type"]).copy()
    anova_2way = two_way_anova(top3_df, factor1_col="a1c_tested", factor2_col="admission_type",
                               value_col="time_in_hospital", alpha=0.05)
    print("\nTwo-Way ANOVA Table (Type II SS):")
    print(anova_2way["anova_table"])
    print("\nEffects Summary (Partial Eta-Squared):")
    print(anova_2way["effects_summary"].to_string(index=False))
    print("\nCell Means (Hospital Stay):")
    print(anova_2way["cell_stats"].to_string(index=False))
    
    # 3C. Tukey HSD Post-Hoc
    tukey_res = tukey_hsd_post_hoc(clean_df, group_col="A1Cresult", value_col="time_in_hospital",
                                   target_groups=["Norm", ">7", ">8"], alpha=0.05)
    print("\nTukey HSD Post-Hoc Comparisons (alpha=0.05):")
    print(tukey_res["comparisons_table"].to_string(index=False))
    
    # Plots for Task 3
    p4 = os.path.join(viz_dir, "04_anova_one_way_distributions.png")
    plot_anova_distributions(clean_df, group_col="A1Cresult", value_col="time_in_hospital",
                             target_groups=["Norm", ">7", ">8"], save_path=p4)
    print(f"Saved: {p4}")
    
    p5 = os.path.join(viz_dir, "05_anova_two_way_interaction.png")
    plot_two_way_anova_interaction(top3_df, factor1_col="a1c_tested", factor2_col="admission_type",
                                   value_col="time_in_hospital", save_path=p5)
    print(f"Saved: {p5}")
    
    p6 = os.path.join(viz_dir, "06_tukey_hsd_forest_plot.png")
    plot_tukey_hsd_results(tukey_res["comparisons_table"], save_path=p6)
    print(f"Saved: {p6}")
    
    p7 = os.path.join(viz_dir, "07_anova_model_diagnostics.png")
    plot_anova_diagnostics(anova_1way["residuals"], anova_1way["fitted_values"],
                           model_name="One-Way ANOVA", save_path=p7)
    print(f"Saved: {p7}")
    
    print("\nPipeline execution complete! All 7 plots generated and statistical tests validated.")


if __name__ == "__main__":
    main()
