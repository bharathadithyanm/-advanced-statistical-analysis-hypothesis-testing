# Advanced Statistical Analysis & Hypothesis Testing
## Inpatient Glycated Hemoglobin ($\text{HbA}_{1c}$) Testing & Hospital Stay Duration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Statistical Rigor](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](tests/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)](LICENSE)

An end-to-end, production-grade clinical biostatistical research project evaluating acute hospital length of stay (`time_in_hospital`, integer days) among adult diabetic inpatients across 130 US hospitals ($N = 101,763$). The project implements formal distribution normality testing, parametric versus non-parametric two-sample hypothesis testing, One-Way and Two-Way Factorial ANOVA with interaction effects, model assumption diagnostics, and Tukey HSD post-hoc multiple comparison testing with Family-Wise Error Rate (FWER) control.

---

## 📊 Summary of Statistical Findings

| Analytical Stage | Statistical Test | Test Statistic | $p$-value | Effect Size | Practical Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normality** | Shapiro-Wilk ($N=5,000$) | $W = 0.8910$ | $1.14 \times 10^{-50}$ | Skewness: $+1.134$ | Departures from continuous Gaussianity; discrete right-skewed count structure. |
| **Normality** | Kolmogorov-Smirnov ($N=101,763$) | $D = 0.1729$ | $< 1.0 \times 10^{-100}$ | Max ECDF diff: $17.3\%$ | Normality rejected; Central Limit Theorem guarantees asymptotic normality of sample means. |
| **Homoscedasticity** | Levene Test (Testing Status) | $F = 79.74$ | $4.34 \times 10^{-19}$ | $s_1^2 = 9.69, s_2^2 = 8.71$ | Variances differ significantly; Welch's unpooled $t$-test strictly required. |
| **Two-Sample Mean** | Welch's Independent $t$-Test | $t(23559.7) = 20.19$ | $7.16 \times 10^{-90}$ | $\Delta = +0.523$ d<br>Cohen's $d = 0.176$ | Tested cohort stays significantly longer ($4.83$ vs $4.31$ d; 95% CI: $[0.47, 0.57]$ d). Small clinical effect. |
| **Two-Sample Rank** | Mann-Whitney $U$ Test | $U = 796,990,222.0$<br>$Z = 21.89$ | $3.25 \times 10^{-106}$ | $r_{\text{rb}} = -0.105$<br>$r = 0.069$ | Medians are identical ($4.0$ d); tested patients exhibit upper-quartile rank shift ($75\text{th}$: $7$ vs $6$ d). |
| **One-Way ANOVA** | Omnibus $F$ (A1C Strata) | $F(2, 17015) = 5.73$ | $0.00325$ | $\eta^2 = 0.0007$<br>$\omega^2 = 0.0006$ | Significant stay differences across tested strata (`Norm`, `>7`, `>8`). Homoscedastic ($p = 0.32$). |
| **Two-Way ANOVA** | Factorial Main & Interaction | Interaction $F = 81.96$ | $2.73 \times 10^{-36}$ | $\eta_p^2 = 0.00179$ | Testing effect is heavily moderated by admission urgency: strongest in Elective admissions ($+1.36$ d). |
| **Tukey HSD** | Post-Hoc: `>8` vs `Norm` | Diff $= +0.175$ d | $0.0050$ | 95% CI: $[0.044, 0.306]$ | Normal A1C patients stay longer than poorly controlled (>8); acute non-glycemic severity in tested normoglycemics. |

---

## 📁 Repository Structure

```text
advanced-statistical-analysis/
├── data/
│   ├── diabetic_data.csv          # Reused primary research cohort (101,766 raw records)
│   └── IDS_mapping.csv            # Standard admission/discharge metadata crosswalk
├── notebooks/
│   └── advanced_statistical_analysis.ipynb # Fully executed 14-section reproducible notebook
├── src/
│   ├── __init__.py                # Package namespace exports
│   ├── normality_tests.py         # Shapiro-Wilk, repeated subsampling, KS test, moments
│   ├── hypothesis_testing.py      # Welch t-test, Mann-Whitney U, Levene test, effect sizes
│   ├── anova_analysis.py          # One-Way ANOVA, Two-Way Factorial ANOVA, Tukey HSD
│   ├── visualization.py           # Publication-grade plotting module (300 DPI Agg backend)
│   ├── run_analysis_pipeline.py   # Standalone end-to-end execution script
│   └── generate_executed_notebook.py # In-process notebook generator with embedded plots
├── visualizations/
│   ├── 01_distribution_overview.png           # 4-panel histogram, KDE, Q-Q, box, violin
│   ├── 02_normality_diagnostics.png           # Focused empirical density vs theoretical normal
│   ├── 03_hypothesis_testing_comparison.png   # 3-panel box, violin, and mean CI plot
│   ├── 04_anova_one_way_distributions.png     # Stratified boxplot and violin across A1C levels
│   ├── 05_anova_two_way_interaction.png       # Factorial interaction plots and marginal means
│   ├── 06_tukey_hsd_forest_plot.png           # Forest plot of pairwise differences & 95% CIs
│   └── 07_anova_model_diagnostics.png         # Residuals vs fitted and Q-Q diagnostics
├── reports/
│   └── statistical_findings.md    # Comprehensive scientific biostatistical findings report
├── tests/
│   ├── __init__.py
│   ├── test_normality_tests.py    # Unit tests for distribution and normality suite
│   ├── test_hypothesis_testing.py # Unit tests for Welch t-test, Mann-Whitney, Levene
│   ├── test_anova_analysis.py     # Unit tests for One-Way/Two-Way ANOVA and Tukey HSD
│   └── test_visualization.py      # Unit tests verifying figure generation
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Pinned dependencies
└── README.md                      # Project documentation and execution instructions
```

---

## 🚀 Quickstart & Execution Instructions

### 1. Environment Setup
Create and activate an isolated virtual environment using Python 3.10+:

```bash
# Navigate to project directory
cd advanced-statistical-analysis

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Or on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Automated Unit Test Suite
Verify that all statistical calculations, distribution functions, effect sizes, and visual modules function properly:

```bash
python -m pytest tests -v
```
*Expected result: 21 passed in ~18 seconds.*

### 3. Run the Complete Analysis Pipeline
Execute the full statistical workflow and generate all publication-grade plots into `visualizations/`:

```bash
python src/run_analysis_pipeline.py
```

### 4. Launch the Reproducible Jupyter Notebook
Open and explore the comprehensive 14-section interactive notebook:

```bash
jupyter notebook notebooks/advanced_statistical_analysis.ipynb
```

The notebook is pre-executed with all cell outputs, tables, and high-resolution base64 plots saved inline.

---

## 📈 Visualizations Overview

All figures are rendered at 300 DPI using a headless `Agg` backend:
1. `visualizations/01_distribution_overview.png`: Combined 4-panel overview: Empirical histogram with KDE, fitted Gaussian overlay, Quantile-Quantile (Q-Q) plot, box plot with summary moments, and kernel density violin plot.
2. `visualizations/02_normality_diagnostics.png`: High-resolution comparative normality diagnostic plot displaying empirical data density against a fitted normal curve and normal Q-Q plot.
3. `visualizations/03_hypothesis_testing_comparison.png`: 3-panel comparative layout: Side-by-side boxplot, split violin plot, and point-and-whisker mean plot with 95% confidence intervals and Cohen's $d$ annotation.
4. `visualizations/04_anova_one_way_distributions.png`: Boxplot and violin distribution plots comparing hospital stay across HbA1c result categories (`Norm`, `>7`, `>8`).
5. `visualizations/05_anova_two_way_interaction.png`: Dual interaction plots illustrating mean hospital stay across testing status stratified by admission urgency, highlighting non-parallel slopes.
6. `visualizations/06_tukey_hsd_forest_plot.png`: Publication forest plot displaying Tukey HSD pairwise mean differences, 95% simultaneous confidence intervals, zero reference line, and decision color coding.
7. `visualizations/07_anova_model_diagnostics.png`: Diagnostic panel showing Residuals vs Fitted values (homoscedasticity check) and Residual Q-Q plot (normality check).

---

## 🔬 Methodological & Epidemiological Guidelines

- **Statistical Significance vs Clinical Magnitude**: In cohorts exceeding 100,000 encounters, microscopic differences attain extreme statistical significance ($p < 10^{-50}$). Researchers must evaluate effect sizes (Cohen's $d = 0.176$, rank-biserial $r = -0.105$) to avoid over-interpreting clinically minor variations.
- **Absence of Proof of Causation**: Association does not imply causation. Inpatient HbA1c testing is ordered per physician judgment; testing status functions as an operational marker for clinical complexity and inpatient care coordination.
- **FWER Control**: Pairwise post-hoc comparisons must always utilize formal multiplicity corrections (e.g. Tukey HSD) to prevent false discovery inflation.

---

## 📜 License
This project is licensed under the Apache License, Version 2.0.
