# Stage 5: Repository Audit & Implementation Plan

**Target System:** Explainable Longitudinal Depression Severity Prediction & Clinical Decision Support Framework  
**Date:** October 2026  
**Auditor / Architect:** Antigravity System  
**Document ID:** STAGE5-AUDIT-01  

---

## 1. Executive Summary & Objective

The objective of Stage 5 is to incorporate the most valuable clinical decision-support and machine learning capabilities from peer implementations (`dep_det_pushti` and `Depression Detection`) into our established longitudinal framework (`d:/projects/DepDet/project`).

Specifically, Stage 5 integrates:
1. **Strong Tabular Benchmark:** A rigorous, subject-level XGBoost multiclass model benchmarked directly against our Stage 4 selected recurrent model ($W=1$ Windowed Fusion GRU).
2. **Two Explicit Clinical Prediction Policies:**
   - **Policy A (Sensor-Only):** Pure passive sensing, digital biomarkers, and static clinical comorbidities (no previous or current PHQ-9).
   - **Policy B (Prior-PHQ-9-Assisted):** Conditioned on the patient's genuinely observed antecedent clinical questionnaire score ($y_{t-1}$).
3. **Severe-Class Operating Decision Threshold Optimization:** Post-hoc probability threshold tuning ($T_4$) on the validation split to systematically address minority severe depression triage without arbitrary heuristic thresholds.
4. **Probability Calibration & Uncertainty Estimation:** Platt (sigmoid) and isotonic calibration evaluated via multiclass Brier score and reliability diagnostics.
5. **TreeSHAP Explainability:** Fast, exact TreeSHAP attributions for tree models alongside permutation feature importance for recurrent models.
6. **Continuous Score Regression vs. Direct Classification:** Empirical comparison of direct continuous score prediction + clinical discretization against discrete classification and multi-task representation.
7. **Interactive Clinical Research Dashboard:** A standalone, dependency-free interactive HTML/JS application displaying metrics, threshold curves, SHAP explanations, and patient trajectories from static JSON artifacts.
8. **Automated Unit Verification:** A dedicated Stage 5 test suite verifying zero leakage, temporal cutoffs, threshold integrity, and dashboard data provenance.

---

## 2. Audit of Existing Repository Capabilities

### 2.1 Capabilities to Reuse
Our existing framework provides a production-grade scientific foundation:
- **`project/src/data_loader.py`:** Standardized ingestion of `anon_processed_df_parquet`, parsing of `user_id` and `period_num`, and clean feature taxonomy (150 predictor columns).
- **`project/src/splitting.py` & `participant_splits.json`:** Fixed subject-level split partitioning 4,948 participants (10,866 labeled assessments) into Train (3,463 participants / 7,694 labeled), Validation (742 participants / 1,597 labeled), and Test (743 participants / 1,575 labeled) with zero participant overlap.
- **`project/src/preprocessing.py`:** Leakage-free preprocessing pipelines fitting imputers and standard scalers strictly on training participant records.
- **`project/src/evaluation.py`:** Standardized 5-class evaluation suite: Accuracy, Balanced Accuracy, Macro-F1, Quadratic Weighted Kappa (QWK), Multiclass G-Mean (penalizing zero-recall classes), and Ordinal/Regression MAE.
- **`project/src/xgboost_model.py`:** Existing XGBoost wrapper with validation-loss early stopping and inverse-frequency sample weighting.
- **`project/src/stage4_progression.py`:** Windowed recurrent tensor extraction, multi-output progression model (GRU + Dense MLP, 5-class logits + continuous PHQ-9 score), within-subject transition tracking ($\Delta y$), and Minimal Clinically Important Difference (MCID $\ge 5$ points) logic.
- **`project/src/test_pipeline_and_leakage.py` & `test_stage4_progression.py`:** 41 passing automated unit tests guaranteeing zero data leakage and mathematical progression validity.

### 2.2 Missing Capabilities to Implement in Stage 5
- Post-hoc probability threshold calibration for the severe depression class ($T_4$).
- Policy B sequence and tabular feature builder that links $y_{t-1}$ strictly chronologically.
- Probability calibration (Platt scaling / Isotonic regression) and Brier score evaluation.
- TreeSHAP integration for XGBoost with domain aggregation and local case waterfalls.
- Standalone interactive HTML/JS dashboard reading static experiment artifacts.
- Comprehensive Stage 5 test suite (`test_stage5_enhancements.py`).

### 2.3 Duplicate Implementations to Avoid
- **Do not recreate split files:** Reuse the frozen `project/data/participant_splits.json`.
- **Do not rewrite data loading:** Reuse `data_loader.py`.
- **Do not overwrite Stage 4 results:** All Stage 5 outputs must be saved strictly under `project/results/stage5_model_enhancements/`.

---

## 3. Discrepancy Audit: Why External Models' Metrics are Non-Comparable

A critical finding from our cross-codebase audit is that metrics reported in `dep_det_pushti` and `Depression Detection` cannot be directly compared against our Stage 4 results without standardization:

| Dimension | `dep_det_pushti` | `Depression Detection` | Our Project (`project`) | Scientific Impact on Comparability |
| :--- | :--- | :--- | :--- | :--- |
| **Split Strategy** | 60.5% Train-Dev ($N=6,578$), 19.5% Val ($N=2,115$), 20.0% Test ($N=2,173$). | 5-Fold StratifiedGroupKFold pooled across all 10,866 rows. | 70.8% Train ($N=7,694$), 14.7% Val ($N=1,597$), 14.5% Test ($N=1,575$). | Different evaluation sample sizes and test cohorts mean raw metric values cannot be juxtaposed directly. |
| **Prior PHQ-9 Handling** | Static sensor-only in Phase 2B; prior survey was planned for Phase 3. | Included `phq9_score_start` as an input feature in general models (Policy B). | Quarantined in Stage 4; explicitly separated into Policy A and Policy B in Stage 5. | Feeding prior score inflates accuracy from ~44% to ~55%; comparing Policy B to Policy A as equal is deceptive. |
| **Observation Window Realism** | Static cross-sectional only (no windowing). | Labeled sessions $s-1, s-2, s-3$ as "10-day, 20-day, 30-day" windows. | Empirically grounded monthly windows ($W=1$ ~30d, $W=2$ ~60d, $W=3$ ~90d, $W=\text{Full}$). | The "10-day" window claim is empirically false in monthly aggregated data. |
| **Threshold $T_4$ Value** | Tuned $T_4^* = 0.14$ on its specific validation split ($N=2,115$). | Did not optimize operating thresholds. | Must sweep and tune $T_4^*$ on our own validation split ($N=1,597$). | Copying $T_4=0.14$ directly would violate scientific discipline; threshold must be re-derived. |
| **Progression Metric** | Not implemented. | Arbitrary $\pm 3$-point delta. | Psychometrically validated MCID ($\ge 5$ points, Kroenke et al.). | Different clinical target definitions prevent cross-codebase comparison. |

---

## 4. Work Package Implementation Plan & File Architecture

All Stage 5 enhancements will be implemented modularly, with zero modification to historical Stage 1–4 artifacts:

```
project/
├── src/
│   ├── stage5_benchmarks.py              # WP2, WP3, WP5, WP7: XGBoost, Dual Policies, Calibration, Regression
│   ├── stage5_threshold_tuning.py        # WP4: Validation-only severe class threshold optimization
│   ├── stage5_explainability.py          # WP6: TreeSHAP, domain aggregation, local waterfalls
│   ├── stage5_dashboard_generator.py     # WP9: Generates standalone interactive dashboard & JSON data
│   ├── run_stage5_pipeline.py            # Master runner orchestrating all Stage 5 experiments
│   └── test_stage5_enhancements.py       # WP10: 16+ unit tests verifying Stage 5 integrity
├── results/
│   └── stage5_model_enhancements/        # All Stage 5 output CSVs, JSONs, plots, and checkpoints
│       ├── stage5_benchmark_comparison.csv
│       ├── stage5_policy_comparison.csv
│       ├── stage5_threshold_optimization.json
│       ├── stage5_calibration_results.json
│       ├── stage5_shap_feature_importance.csv
│       ├── plots/
│       └── dashboard/                    # WP9 Standalone Research Dashboard
│           ├── index.html
│           ├── dashboard_data.json
│           └── dashboard.css / .js
```

### Proposed Execution Sequence:
1. **Step 1:** Implement `project/src/stage5_benchmarks.py` covering:
   - XGBoost classifier on identical features.
   - Policy A (Sensor-Only) vs. Policy B (Prior-PHQ-9-Assisted with strictly chronological $y_{t-1}$).
   - Direct Continuous Score Regression with clinical discretization.
   - Platt & Isotonic probability calibration evaluated via Brier score and log loss.
2. **Step 2:** Implement `project/src/stage5_threshold_tuning.py`:
   - Sweep $T_4 \in [0.05, 0.50]$ on Validation split.
   - Optimize clinical utility objective: $\mathcal{J}_{\text{val}} = \text{Macro-F1}_{\text{val}} + 0.50 \cdot F_{1,\text{val}}^{(4)}$ with guardrails.
   - Freeze threshold and evaluate once on held-out test set.
3. **Step 3:** Implement `project/src/stage5_explainability.py`:
   - TreeSHAP on XGBoost for global beeswarm, domain aggregation, and local severe case explanations.
4. **Step 4:** Implement `project/src/stage5_dashboard_generator.py`:
   - Export standalone, interactive `index.html` loading frozen experiment artifacts.
5. **Step 5:** Create and run `project/src/test_stage5_enhancements.py`:
   - Verify subject isolation, temporal cutoffs, prior-score alignment, threshold valid predictions, and dashboard loading without retraining.
6. **Step 6:** Run `run_stage5_pipeline.py` to generate all experimental artifacts.
7. **Step 7:** Compile `progress_reports/STAGE5_EXPERIMENT_REPORT.md`.

---

**Audit Approval Status:** Approved. Ready to execute Work Package 2 through Work Package 11.
