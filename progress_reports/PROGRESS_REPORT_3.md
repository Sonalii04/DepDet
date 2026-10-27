# 📊 Progress Report 3: Controlled Benchmark Experiments, Longitudinal Evaluation & Multimodal Analysis

- **Report ID:** PR-03 (Tabular & Structured Edition)
- **Date:** October 2026
- **Execution Script:** [`project/src/run_stage3_benchmark.py`](file:///d:/projects/DepDet/project/src/run_stage3_benchmark.py)
- **Unit Test Suite:** [`project/src/test_pipeline_and_leakage.py`](file:///d:/projects/DepDet/project/src/test_pipeline_and_leakage.py) (26/26 tests passing)
- **Status:** Completed & Empirically Verified across all models, matched cohorts, feature ablations, and multi-seed runs.

---

## 1. Executive Summary & Selection Winners

### Primary Selection Criterion
Under strict scientific protocol, **Validation Macro-F1** was prospectively defined as the sole primary model selection criterion. The test set ($N=1,575$ labeled assessments across 602 participants) remained untouched during hyperparameter tuning, model choice, and checkpoint selection.

### Key Performance Comparison (Selected Champions)

| Evaluation Profile | Best Model | Primary Metric Score | Test Macro-F1 | Test Balanced Acc | Test Accuracy | Test QWK | Clinical / Operational Strength |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Primary Selection Winner** | **Multi-Task Aux-Reg MLP ($\lambda=0.5$)** | **Val Macro-F1: 0.3427** | 0.3150 | 0.3127 | 45.27% | 0.4540 | Joint continuous-discrete calibration; lowest ordinal error. |
| **Minority Screening Winner** | **Class-Weighted MLP** | **Val Bal-Acc: 0.3625** | 0.3184 | **0.3534** | 40.95% | 0.4553 | Highest severe-class sensitivity (**40.28% recall** vs. 20.8% baseline). |
| **Ordinal Consistency Winner** | **Frank–Hall Ordinal Classifier** | **Val Acc: 44.77%** | 0.3283 | 0.3226 | **45.52%** | 0.4351 | Enforces monotonic cumulative clinical severity boundaries. |
| **Top Descriptive Test F1** | **Longitudinal GRU (Current Only)** | Val Macro-F1: 0.3126 | **0.3437** | 0.3389 | 45.33% | **0.4607** | Captures acute monthly sensor dynamics ($W=1$). |

---

## 2. Complete Controlled Benchmark Results

All models were evaluated under identical subject-level partitioning using training-only preprocessing, training-only class weighting/resampling, and fixed class ordering `[0, 1, 2, 3, 4]`.

### Data Partitioning & Evaluation Denominators

| Split | Total Passive Sensor Rows | Unique Tracking Participants | Labeled PHQ-9 Assessments ($N_{\text{eval}}$) | Labeled Participants | Target Data Availability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Split** | 25,277 | 3,463 | **7,694** | 2,830 | 100% training-only fitting & preprocessing |
| **Validation Split** | 5,217 | 742 | **1,597** | 604 | Dedicated hyperparameter & window selection |
| **Test Split** | 5,200 | 743 | **1,575** | 602 | Held-out evaluation (evaluated once blind) |
| **Full Study Total** | **35,694** | **4,948** | **10,866** | **4,036** | Strict zero-leakage participant partitioning |

> *All benchmark metrics below evaluate exclusively on non-missing labeled assessments ($N_{\text{val}} = 1,597$; $N_{\text{test}} = 1,575$).*

### Benchmark Results Table (Validation Selection vs. Held-Out Test Evaluation)

| # | Model / Architecture | Model Category | Val Macro-F1 | Val Bal-Acc | Val Acc (%) | Val G-Mean | Test Macro-F1 | Test Bal-Acc | Test Acc (%) | Test G-Mean | Test QWK | Test Ordinal MAE | Test Score MAE | Test ROC-AUC |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Logistic Regression (L2) | Linear Baseline | 0.3108 | 0.3117 | 44.33% | 0.2276 | 0.3257 | 0.3212 | 45.40% | 0.2563 | 0.4238 | 0.7797 | — | 0.6978 |
| **2** | Unweighted Classification MLP | Neural Baseline | 0.3274 | 0.3258 | 44.02% | 0.2688 | 0.3116 | 0.3156 | 44.25% | 0.2457 | 0.4256 | 0.8025 | — | — |
| **3** | Class-Weighted MLP | Imbalance Strategy | 0.3090 | **0.3625** | 38.82% | **0.3209** | 0.3184 | **0.3534** | 40.95% | **0.3212** | 0.4553 | 0.9384 | — | — |
| **4** | Corrected Latent-SMOTE MLP | Imbalance Strategy | 0.3206 | 0.3472 | 39.64% | 0.3224 | 0.3068 | 0.3311 | 38.16% | 0.3042 | 0.4019 | 0.9632 | — | 0.6589 |
| **5** | HistGradientBoosting | Tabular Tree | 0.2976 | 0.3005 | 44.83% | 0.2018 | 0.3272 | 0.3176 | 45.08% | 0.2527 | 0.4063 | 0.7797 | — | 0.6961 |
| **6a**| XGBoost (Unweighted) | Tabular Tree | 0.2816 | 0.2881 | 44.21% | 0.1676 | 0.2914 | 0.2958 | 45.40% | 0.1929 | 0.3745 | 0.7898 | — | **0.7028** |
| **6b**| XGBoost (Class-Weighted) | Imbalance Strategy | 0.3216 | 0.3262 | 41.45% | 0.2935 | 0.3128 | 0.3182 | 42.03% | 0.2793 | **0.4625** | 0.8311 | — | 0.6877 |
| **7** | Frank–Hall Ordinal Classifier | Ordinal Decomp. | 0.3231 | 0.3195 | 44.77% | 0.2442 | 0.3283 | 0.3226 | **45.52%** | 0.2582 | 0.4351 | 0.7702 | — | 0.6949 |
| **8** | Longitudinal GRU (Current Only)| Recurrent Neural | 0.3126 | 0.3144 | 43.77% | 0.2431 | **0.3437** | 0.3389 | 45.33% | 0.2907 | 0.4607 | 0.7733 | 3.9860 | — |
| **9** | Longitudinal GRU (History Only)| Recurrent Neural | 0.3007 | 0.2966 | 42.20% | 0.2207 | 0.3121 | 0.3012 | 40.84% | 0.2569 | 0.4257 | 0.7925 | 4.3563 | — |
| **10**| Longitudinal GRU (Fusion) | Recurrent Fusion | 0.3196 | 0.3198 | 45.02% | 0.2367 | 0.3037 | 0.3089 | 44.44% | 0.1769 | 0.4302 | **0.7676** | 4.0191 | — |
| **11**| Multi-Task Aux-Reg MLP ($w=0.5$)| Multi-Task Regular.| **0.3427** | 0.3326 | **45.08%** | 0.2708 | 0.3150 | 0.3127 | 45.27% | 0.2273 | 0.4540 | 0.7479 | **3.9413** | — |

*Artifact Reference:* Saved in [`project/results/metrics/stage3_benchmark_results.csv`](file:///d:/projects/DepDet/project/results/metrics/stage3_benchmark_results.csv) and [`stage3_benchmark_results.json`](file:///d:/projects/DepDet/project/results/metrics/stage3_benchmark_results.json).

---

## 3. Fair Longitudinal Comparison (Matched vs. Full Cohort)

Because **History-Only** cannot generate predictions at baseline ($t=1$, where no preceding history exists), an unbiased comparison requires evaluating on the **identical matched subset of assessments with available antecedent history ($t \ge 2$, `history_len >= 1`)**.

### Longitudinal Evaluation Cohort Reconciliation

| Longitudinal Subcohort | $N$ Assessments | $N$ Participants | History Availability | Evaluation Role |
| :--- | :---: | :---: | :--- | :--- |
| **Full Labeled Test Cohort** | 1,575 | 602 | All labeled test observations ($t \ge 1$) | Unconstrained benchmark evaluation |
| **Verified Matched Cohort ($t \ge 2$)** | **1,523** | **578** | $\ge 1$ preceding period of tracking data | Fair head-to-head comparison across all models |
| **Baseline-Only Observations ($i = 0$)**| 52 | 52 | Initial observation only (`history_len == 0`) | Ineligible for history-dependent models |

### Matched vs. Full Cohort Performance Table

| Model Architecture | Evaluation Cohort | $N$ Assessments | $N$ Participants | Test Macro-F1 | Test Bal-Acc | Test Acc (%) | Test G-Mean | Test QWK | Test Ordinal MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Longitudinal Current Only** | **Matched ($t \ge 2$)** | **1,523** | **578** | **0.3472** | **0.3411** | **45.44%** | **0.2944** | **0.4585** | 0.7708 |
| **Longitudinal History Only** | **Matched ($t \ge 2$)** | **1,523** | **578** | 0.3121 | 0.3012 | 40.84% | 0.2569 | 0.4257 | 0.7925 |
| **Longitudinal GRU Fusion** | **Matched ($t \ge 2$)** | **1,523** | **578** | 0.3052 | 0.3101 | 44.58% | 0.1789 | 0.4291 | **0.7656** |
| *Longitudinal Current Only* | *Full Cohort (All $t$)* | 1,575 | 602 | 0.3437 | 0.3389 | 45.33% | 0.2907 | 0.4607 | 0.7733 |
| *Longitudinal GRU Fusion* | *Full Cohort (All $t$)* | 1,575 | 602 | 0.3037 | 0.3089 | 44.44% | 0.1769 | 0.4302 | 0.7676 |

### Core Longitudinal Takeaways:
1. **Concurrent State Dominates:** History-Only achieves **0.3121 Macro-F1** vs. **0.3472 Macro-F1** for Current-Only on identical matched data ($-0.0351$ penalty for excluding concurrent state).
2. **Distant History Does Not Boost Macro-F1:** Adding long-term GRU history to current features yielded 0.3052 vs. 0.3472, but slightly lowered Ordinal MAE (0.7656 vs. 0.7708).

*Artifact Reference:* Saved in [`project/results/metrics/stage3_matched_longitudinal_results.csv`](file:///d:/projects/DepDet/project/results/metrics/stage3_matched_longitudinal_results.csv) and [`longitudinal_matched_comparison.png`](file:///d:/projects/DepDet/project/results/plots/stage3/longitudinal_matched_comparison.png).

---

## 4. Feature-Group Ablations

The 150 predictor features were cleanly partitioned into distinct clinical and digital phenotype modalities without overlap:

### Feature Modality Definitions

| Modality Name | Feature Count ($D$) | Key Representative Features Included | Modality Scope |
| :--- | :---: | :--- | :--- |
| **Actigraphy / Movement** | 46 | `steps_*`, `mvpa_*`, `lpa_*`, `sedentary_*`, `active_day_*` | Objective physical activity, sedentary bouts |
| **Sleep / Circadian** | 65 | `sleep_*`, `bed_*`, `hyposomnia_*`, `circadian_*`, `asleep_*` | Sleep architecture, fragmentation, onset timing |
| **Demographics** | 18 | `sex`, `birthyear`, `educ`, `height`, `weight`, `money*`, `household` | Baseline biological and socioeconomic status |
| **Clinical Comorbidities** | 21 | `comorbid_*`, `med_*`, `nonmed_*`, `life_*`, `num_migraine_days` | Chronic medical history, migraines, life stress |
| **Combined Modalities** | **150** | All 150 feature columns combined | Multimodal fusion representation |

### Ablation Performance Benchmark Table

| Evaluated Feature Subset | Feature Count | Val Macro-F1 | Val Bal-Acc | Val Acc (%) | Test Macro-F1 | Test Bal-Acc | Test Acc (%) | Test G-Mean | Test QWK | Test Ordinal MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Actigraphy Only** | 46 | 0.1923 | 0.2276 | 38.82% | 0.1853 | 0.2203 | 39.37% | 0.0000 | 0.1118 | 0.9657 |
| **Sleep Only** | 65 | 0.2030 | 0.2282 | 37.32% | 0.2050 | 0.2305 | 39.68% | 0.0930 | 0.1405 | 0.9695 |
| **Demographics Only** | 18 | 0.2448 | 0.2636 | 40.20% | 0.2452 | 0.2571 | 41.08% | 0.1533 | 0.2265 | 0.9187 |
| **Clinical Comorbidities Only** | 21 | 0.2490 | 0.2699 | 43.83% | 0.2837 | 0.2871 | 43.87% | 0.1913 | 0.3303 | 0.8248 |
| **Combined Modalities** | **150** | **0.3108** | **0.3117** | **44.33%** | **0.3257** | **0.3212** | **45.40%** | **0.2563** | **0.4238** | **0.7797** |

### Key Ablation Insights:
- **Clinical features are the strongest standalone predictor:** Attains **0.2837 Test Macro-F1**, substantially exceeding Sleep alone (0.2050) and Actigraphy alone (0.1853).
- **Multimodal synergy:** Combining digital sensors with clinical comorbidities provides an absolute gain of **+0.0420 Macro-F1 (+14.8%)** over clinical features alone, driving QWK to **0.4238** and Ordinal MAE down to **0.7797**.

*Artifact Reference:* Saved in [`project/results/metrics/stage3_ablation_results.csv`](file:///d:/projects/DepDet/project/results/metrics/stage3_ablation_results.csv) and [`feature_group_ablation.png`](file:///d:/projects/DepDet/project/results/plots/stage3/feature_group_ablation.png).

---

## 5. Multi-Seed Stability Analysis

To verify algorithmic stability under distinct random initializations, candidate architectures were tested across **3 seeds (42, 123, 2026)** on the fixed subject split:

| Model Architecture | Test Macro-F1 (Mean $\pm$ Std) | Test Bal-Acc (Mean $\pm$ Std) | Test Accuracy (Mean $\pm$ Std) | Algorithmic Stability Profile |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | $0.3257 \pm 0.0000$ | $0.3212 \pm 0.0000$ | $45.40\% \pm 0.00\%$ | **Deterministic** (Convex L-BFGS solver) |
| **XGBoost (Class-Weighted)** | $0.3124 \pm 0.0056$ | $0.3256 \pm 0.0049$ | $41.52\% \pm 0.72\%$ | **High Stability** ($\sigma_{\text{F1}} = 0.0056$) |
| **Class-Weighted MLP** | $0.3077 \pm 0.0097$ | $\mathbf{0.3448 \pm 0.0082}$ | $39.41\% \pm 1.40\%$ | **Stable Minority Sensitivity** ($\text{Bal-Acc} > 0.34$) |
| **Unweighted MLP** | $0.3119 \pm 0.0138$ | $0.3099 \pm 0.0119$ | $43.17\% \pm 1.50\%$ | **Moderate Variance** ($\sigma_{\text{F1}} = 0.0138$) |

*Artifact Reference:* Saved in [`project/results/metrics/stage3_multiseed_summary.csv`](file:///d:/projects/DepDet/project/results/metrics/stage3_multiseed_summary.csv).

---

## 6. Explainability, Interpretability & Visualizations

| Figure Artifact | File Path | Focus & Architecture | Clinical & Manifold Takeaway |
| :--- | :--- | :--- | :--- |
| **Confusion Matrices** | [`confusion_matrices_comparison.png`](file:///d:/projects/DepDet/project/results/plots/stage3/confusion_matrices_comparison.png) | Unweighted vs. Weighted vs. Frank–Hall | Class weighting drastically cuts severe false negatives (recall 20.8% $\to$ 40.3%). |
| **Per-Class F1 Profile** | [`per_class_f1_comparison.png`](file:///d:/projects/DepDet/project/results/plots/stage3/per_class_f1_comparison.png) | F1 across all 5 clinical tiers | Class 0 attains high F1 ($\approx 0.64$); classes 2–4 require weighted/ordinal penalties. |
| **Feature Ablation Curve** | [`feature_group_ablation.png`](file:///d:/projects/DepDet/project/results/plots/stage3/feature_group_ablation.png) | Modality step progression | Clear synergistic staircase: Actigraphy ($0.185$) $\to$ Clinical ($0.284$) $\to$ Combined ($0.326$). |
| **Longitudinal Matched** | [`longitudinal_matched_comparison.png`](file:///d:/projects/DepDet/project/results/plots/stage3/longitudinal_matched_comparison.png) | Current vs. History vs. Fusion ($N=1,523$) | Demonstrates clear dominance of current observations over historical sequences. |
| **t-SNE Latent Space** | [`tsne_latent_space.png`](file:///d:/projects/DepDet/project/results/plots/stage3/tsne_latent_space.png) | 2D MLP Latent Representations | Continuous clinical gradient from Minimal (purple) to Severe (yellow), confirming ordinal structure. |
| **SHAP Feature Ranking** | [`shap_feature_importance.png`](file:///d:/projects/DepDet/project/results/plots/stage3/shap_feature_importance.png) | TreeExplainer on Class 4 Severe | Top risk factors: `life_stress`, `med_start`, `comorbid_migraines`, and sleep disruption. |

---

## 7. Severe Depression (Class 4) Minority Detection Trade-Offs

True Class 4 (Severe Depression) accounts for only **4.57%** of the held-out test cohort (72 out of 1,575 assessments). The audited confusion matrices ([`stage3_confusion_matrices.json`](file:///d:/projects/DepDet/project/results/metrics/stage3_confusion_matrices.json)) reveal clear operational trade-offs:

### Model-Specific Severe-Class (Class 4) Performance Breakdown

| Model / Architecture | Imbalance / Decision Strategy | True Cases Identified (TP / 72) | Total Flagged (TP + FP) | Severe Recall (Sensitivity) | Severe Precision (PPV) | Clinical Role Recommendation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Class-Weighted MLP** | Inverse-Frequency Loss Penalty | **29 / 72** | 206 | **40.28%** | 14.08% | **Screening Triage:** Maximizes detection of high-risk patients. |
| **Unweighted MLP** | Standard Empirical Loss | 15 / 72 | 67 | 20.83% | 22.39% | Baseline neural benchmark. |
| **Logistic Regression (L2)** | Convex Linear Baseline | 13 / 72 | 43 | 18.06% | 30.23% | Balanced linear comparator. |
| **Frank–Hall Ordinal Classifier** | Cumulative Thresholding | 12 / 72 | 37 | 16.67% | 32.43% | **Conservative Screening:** Enforces ordinal risk monotonicity. |
| **HistGradientBoosting** | Histogram Tree Ensemble | 12 / 72 | 27 | 16.67% | **44.44%** | **High-Confidence Triage:** Minimizes false alarm burden. |
| **Longitudinal Current GRU** | Recurrent Neural ($W=1$) | 12 / 72 | 33 | 16.67% | 36.36% | High precision sequential screening. |
| **XGBoost (Class-Weighted)** | Weighted Sample Gradient Boosting | 14 / 72 | 83 | 19.44% | 16.87% | Intermediate tree sensitivity. |
| **XGBoost (Unweighted)** | Unweighted Tree Ensemble | 9 / 72 | 27 | 12.50% | 33.33% | Low sensitivity minority classification. |
| **Aux-Reg MLP ($w=0.5$)** | Continuous Score Regularization | 4 / 72 | 19 | 5.56% | 21.05% | Continuous score calibration priority. |

---

## 8. Reproducibility Commands

To reproduce all Stage 3 results from scratch:
```bash
# 1. Execute the 26-test unit verification suite
python -m unittest project/src/test_pipeline_and_leakage.py

# 2. Run the complete Stage 3 benchmark and ablation runner
python project/src/run_stage3_benchmark.py
```
Outputs are written with deterministic seeds to `project/results/metrics/` and `project/results/plots/stage3/`.
