# 🧠 Model Architecture & Source Code Directory Map

This directory contains the complete source code implementation of the **Explainable Longitudinal Depression Severity Progression Framework** (Stages 1 through 5).

---

## 📂 Source Code Taxonomy

| File | Primary Role | Pipeline Stage | Description & Key Functions |
| :--- | :--- | :---: | :--- |
| **[`data_loader.py`](data_loader.py)** | Data Ingestion | Core | Loads raw Parquet (`anon_processed_df_parquet`), parses `user_id` and `period_num`, classifies 150 predictor columns across modalities. |
| **[`splitting.py`](splitting.py)** | Participant Partitioning | Core | Enforces subject-level splitting (`participant_splits.json`: 3,463 Train, 742 Val, 743 Test participants; zero overlap guarantee). |
| **[`preprocessing.py`](preprocessing.py)** | Feature Transformation | Core | Leakage-free `DataPreprocessor` fitting `SimpleImputer` and `StandardScaler` strictly on training data. |
| **[`evaluation.py`](evaluation.py)** | Standardized Metrics | Core | Multi-class metrics (Accuracy, Balanced Accuracy, Macro-F1, Quadratic Weighted Kappa [QWK], Multiclass G-Mean with zero-recall penalty, Ordinal MAE). |
| **[`stage5_benchmarks.py`](stage5_benchmarks.py)** | Benchmarks & Dual Policies | Stage 5 | Implements Policy A (Sensor-Only), Policy B (Prior-PHQ-9-Assisted), probability calibration (Platt/Isotonic), and continuous score regression. |
| **[`stage5_threshold_tuning.py`](stage5_threshold_tuning.py)** | Severe Class Calibration | Stage 5 | Post-hoc operating decision threshold optimization ($T_4^* = 0.2100$) on validation data to maximize severe depression detection. |
| **[`stage5_explainability.py`](stage5_explainability.py)** | TreeSHAP Explainability | Stage 5 | Exact `shap.TreeExplainer` for XGBoost, feature rankings, domain attribution breakdowns, and local clinical alert case studies. |
| **[`stage5_dashboard_generator.py`](stage5_dashboard_generator.py)** | Research Dashboard | Stage 5 | Generates standalone, dependency-free interactive HTML/CSS/JS application displaying metrics, threshold simulator, and trajectories. |
| **[`stage4_progression.py`](stage4_progression.py)** | Recurrent Fusion GRU | Stage 4 | Windowed tensor sequence builder ($W \in \{1, 2, 3, \text{Full}\}$), multi-output progression model (GRU + MLP), within-subject $\Delta y_t$ tracking, and MCID logic. |
| **[`stage4_temporal_audit.py`](stage4_temporal_audit.py)** | Temporal Data Audit | Stage 4 | Verifies dataset temporal structure as Case C (period-level aggregates, quarterly PHQ-9 milestones). |
| **[`stage4_explainability.py`](stage4_explainability.py)** | Temporal Explainability | Stage 4 | Permutation feature importance on continuous PHQ-9 scale and patient trajectory visualizers. |
| **[`xgboost_model.py`](xgboost_model.py)** | XGBoost Wrapper | Stage 3/5 | Multiclass `XGBClassifier` with sample weighting, early stopping on validation loss, and checkpoint management. |
| **[`baseline.py`](baseline.py)** | Static Baselines | Stage 1–3 | Baseline linear and neural classifiers (Logistic Regression, HistGradientBoosting, MLP). |
| **[`multitask.py`](multitask.py)** | Auxiliary Regression | Stage 2–3 | Multi-task neural network combining classification cross-entropy and continuous score auxiliary regression loss ($\lambda = 0.5$). |
| **[`contrastive.py`](contrastive.py)** | Supervised Contrastive | Stage 1 | Supervised contrastive representation learning encoder. |
| **[`oversampling.py`](oversampling.py)** | Latent SMOTE | Stage 1–2 | Corrected latent-space synthetic minority oversampling in encoder representations. |
| **[`data_leakage_checks.py`](data_leakage_checks.py)** | Automated Audit | Stage 1–3 | Automated verification of participant isolation, feature overlap, and temporal bounds. |

---

## 🚀 Execution Runners

| Runner Script | Command | Purpose |
| :--- | :--- | :--- |
| **[`run_stage5_pipeline.py`](run_stage5_pipeline.py)** | `python project/src/run_stage5_pipeline.py` | **Master Stage 5 Runner:** Trains Policy A/B models, tunes $T_4^* = 0.21$, computes calibration & TreeSHAP, and generates dashboard. |
| **[`run_stage4_experiments.py`](run_stage4_experiments.py)** | `python project/src/run_stage4_experiments.py` | **Stage 4 Progression Runner:** Evaluates observation windows ($W=1, 2, 3, \text{Full}$) and Persistence Baseline across 973 transitions. |
| **[`run_stage3_benchmark.py`](run_stage3_benchmark.py)** | `python project/src/run_stage3_benchmark.py` | **Stage 3 Benchmark Grid:** Evaluates 11 candidate architectures on the controlled test split. |

---

## 🧪 Automated Unit Test Suites (50 Tests Passing)

| Test Suite | Command | Number of Tests | Properties Verified |
| :--- | :--- | :---: | :--- |
| **[`test_stage5_enhancements.py`](test_stage5_enhancements.py)** | `python -m unittest project/src/test_stage5_enhancements.py` | **9 Tests** | Zero split overlap, Policy A target quarantine, Policy B chronology, missing prior score handling, valid threshold classes, dashboard integrity. |
| **[`test_stage4_progression.py`](test_stage4_progression.py)** | `python -m unittest project/src/test_stage4_progression.py` | **15 Tests** | Temporal window extraction, GRU tensor dimensions, within-subject transition logic, MCID 3-class boundary arithmetic, persistence baseline. |
| **[`test_pipeline_and_leakage.py`](test_pipeline_and_leakage.py)** | `python -m unittest project/src/test_pipeline_and_leakage.py` | **26 Tests** | Zero participant leakage, training-only preprocessor fitting, metric determinism, Frank–Hall decomposition, feature matrix sanity. |
