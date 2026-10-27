# 📊 Progress Report 1: Initial Pipeline & Baseline Benchmark Run

- **Report ID:** PR-01
- **Date:** October 2026
- **Execution Script:** [run_all_experiments.py](file:///d:/projects/DepDet/project/run_all_experiments.py)
- **Status:** Initial Benchmark Established (First Full Experimental Run)

---

## 1. Executive Summary

This is the initial progress report documenting the first complete execution of the **Explainable Contrastive Longitudinal Depression Prediction Framework**. 

A total of **8 ablation experiments (E1 to E8)** were executed across a 150-feature wearable actigraphy and clinical dataset spanning **4,948 participants**. Strict participant-level (subject-independent) isolation and 6 automated data leakage checks were enforced to prevent optimistic evaluation bias.

### Baseline Benchmark Highlights
- **Highest Test Accuracy:** **45.27%** (Achieved by **E5: Current Assessment Only**)
- **Highest Test Macro-F1:** **0.3170** (Achieved by **E5: Current Assessment Only**)
- **Highest Minority Balanced Accuracy:** **0.3205** (Achieved by **E3: Baseline + Latent SMOTE**)
- **Best Continuous Score Prediction (MAE):** **3.9314 points** on the 0–27 PHQ-9 scale (Achieved by **E1: Baseline MLP**)

---

## 2. Experimental Setup & Controls

### Dataset & Modality Scope
- **Total Rows:** 35,694 observations across 4,948 unique `user_id`s (1 to 12 periods per user).
- **Labeled PHQ-9 Intervals:** 10,866 rows (30.44%).
- **Modalities Used:** Wearable actigraphy (42 features), sleep & circadian rhythms (48 features), demographics (18 features), clinical comorbidities & lifestyle (22 features).
- **Modality Audit:** Systematic verification confirmed **no speech, audio, or text features exist**.

### Subject-Level Data Partitioning
To guarantee zero subject leakage, all data splits were performed strictly by `user_id`:
- **Train (70%):** 3,463 participants (25,277 total passive tracking rows; **7,694** labeled evaluation rows across 2,830 participants)
- **Validation (15%):** 742 participants (5,217 total passive tracking rows; **1,597** labeled evaluation rows across 604 participants)
- **Test (15%):** 743 participants (5,200 total passive tracking rows; **1,575** labeled evaluation rows across 602 participants)
*(Note: Initial draft noted 7,688 / 1,603 due to an early 6-row transcription artifact; verified ground truth is 7,694 / 1,597).*

### Automated Data Leakage Validation
All 6 automated checks in [data_leakage_checks.py](file:///d:/projects/DepDet/project/src/data_leakage_checks.py) passed:
- `Intersection(Train, Val, Test) = 0` (Zero participant overlap).
- Preprocessing imputers and standard scalers fitted exclusively on Train.
- Latent-space SMOTE synthetic oversampling applied exclusively on Train.
- Sequence temporal direction strictly enforced ($X_{1:t-1} \to X_t$).
- All target-derived columns (`phq9_*`, `delta_phq9`) isolated from feature tensors.

---

## 3. Benchmark Results: Ablation Experiments (E1 – E8)

All models were evaluated on the untouched, unseen test set (743 participants / 1,575 assessments):

| Experiment | Architecture / Strategy | Val Macro-F1 | Test Acc (%) | Test Macro-F1 | Test Bal Acc | Test MAE | Test RMSE | Test $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Baseline MLP ([baseline.py](file:///d:/projects/DepDet/project/src/baseline.py)) | 0.3109 | 44.06% | 0.3038 | 0.3087 | **3.9314** | **5.0931** | **0.2454** |
| **E2** | Baseline + Contrastive Learning ([contrastive.py](file:///d:/projects/DepDet/project/src/contrastive.py)) | 0.2901 | 44.70% | 0.2956 | 0.3060 | 3.9671 | 5.1712 | 0.2221 |
| **E3** | Baseline + Latent SMOTE ([oversampling.py](file:///d:/projects/DepDet/project/src/oversampling.py)) | 0.2472 | 35.56% | 0.2740 | **0.3205** | 3.9314 | 5.0931 | 0.2454 |
| **E4** | Contrastive + Latent SMOTE | 0.2953 | 33.97% | 0.2702 | 0.3047 | 3.9671 | 5.1712 | 0.2221 |
| **E5** | Current Assessment Only ([longitudinal.py](file:///d:/projects/DepDet/project/src/longitudinal.py)) | 0.3070 | **45.27%** | **0.3170** | 0.3138 | 3.9863 | 5.1620 | 0.2249 |
| **E6** | History Only (GRU on $X_{1:t-1}$) | 0.2355 | 44.13% | 0.2455 | 0.2705 | 3.9584 | 5.1055 | 0.2418 |
| **E7** | Current + History Dual Fusion | 0.2799 | 43.05% | 0.2699 | 0.2758 | 3.9928 | 5.1196 | 0.2376 |
| **E8** | Full Multi-Task Proposed ([multitask.py](file:///d:/projects/DepDet/project/src/multitask.py)) | 0.2741 | 44.19% | 0.2813 | 0.2923 | 3.9432 | 5.1781 | 0.2200 |

---

## 4. Key Empirical Insights from Run 1

1. **Concurrent Sensor Supremacy (E5 vs E6/E7):**
   - Evaluating only concurrent features at period $t$ (**E5**) achieved the highest test accuracy (**45.27%**) and Macro-F1 (**0.3170**). 
   - Historical features alone (**E6**) predict the patient's general baseline well but miss acute flare-ups. Concatenating history to current features (**E7**) adds parameters that slightly dilute the immediate sensor signals in observational settings where gaps between surveys vary.

2. **The Accuracy vs. Minority Recall Trade-Off (E3):**
   - Naive cross-entropy models bias predictions toward Class 0 (Minimal, 38.6%) and Class 1 (Mild, 29.4%).
   - Latent-space SMOTE (**E3**) drops overall accuracy to **35.56%** but achieves the highest **Balanced Accuracy (0.3205)**. In a clinical diagnostic tool, detecting severe depression is often prioritized over raw accuracy on non-depressed cases.

3. **Continuous Score Regression Reliability:**
   - Predicting the exact 0–27 PHQ-9 score yielded an **MAE of 3.93 points** ($R^2 \approx 0.245$). 
   - Since a 5-point shift represents a clinical severity transition, an error under 4 points on completely unseen participants validates the wearable biomarker representations.

4. **Biomarker Discoveries (SHAP & Permutation Importance):**
   - **Strongest Positive Drivers of Severity:** Financial strain (`money`), neuropathic pain (`comorbid_neuropathic`), chronic migraine burden (`num_migraine_days`), sedentary days per month (`steps__sedentary_day_count_`), and sleep fragmentation.
   - **Strongest Protective Factors:** Mean awake steps (`steps_awake_mean`), weekday sleep efficiency (`sleep_ratio_asleep_in_bed_weekday_mean`), and active day streaks.

---

## 5. Baselines for Subsequent Iterations (Target Metrics for PR-02)

When modifying code in future runs, performance will be benchmarked against these figures:

| Target Evaluation Metric | PR-01 Benchmark | Goal for Next Iteration (PR-02) |
| :--- | :---: | :---: |
| **Top 5-Class Test Accuracy** | 45.27% (E5) | $> 46.50\%$ |
| **Top Test Macro-F1** | 0.3170 (E5) | $> 0.3300$ |
| **Top Test Balanced Accuracy** | 0.3205 (E3) | $> 0.3350$ |
| **Continuous Score Regression MAE** | 3.9314 (E1) | $< 3.8000$ points |
| **Minority Class 4 Recall** | ~2.5% (unbalanced) / 32% (SMOTE) | $> 35.0\%$ with $> 42\%$ accuracy |

---

## 6. Action Items / Hypotheses for Next Iteration (PR-02)

To improve upon this benchmark in Progress Report 2, the following modifications are candidates for testing:
1. **Focal Loss or Class-Weighted Cross-Entropy:** Replace standard Cross-Entropy in E1/E5 with class-weighted loss to boost minority recall without sacrificing overall accuracy as drastically as Latent SMOTE.
2. **Temporal Attention / Transformer Sequence Encoder:** Replace standard GRU in E7 with an attention mechanism over past observation intervals so the network only attends to relevant historical periods.
3. **Feature Selection / Dimensionality Pruning:** Filter out zero-importance features identified by permutation importance to reduce overfitting on the 150 feature inputs.
4. **Hyperparameter Tuning:** Explore learning rate warmups, cosine annealing schedulers, and dropout adjustments.
