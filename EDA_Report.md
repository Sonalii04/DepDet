# 📊 Exploratory Data Analysis (EDA) Report
**Dataset Path:** `D:\projects\DepDet\anon_processed_df_parquet`  
**File Size:** 21.3 MB | **Format:** Parquet  
**Dimensions:** 35,694 Rows × 154 Columns  
**Subjects:** 4,948 Unique Participants (`user_id`)  

---

## Executive Summary

This report presents a thorough Exploratory Data Analysis (EDA) of the `anon_processed_df_parquet` dataset, designed for digital depression detection using longitudinal wearable sensor actigraphy, circadian sleep dynamics, clinical baseline surveys, demographic profiles, and medical comorbidities.

### Key Insights & Biomarkers Discovered

1. **Longitudinal Cohort Structure:**
   - The dataset tracks **4,948 unique individuals** across up to **12 observation periods** (average 7.2 periods per user).
   - **10,866 observation intervals (30.4%)** contain matched PHQ-9 survey assessments, while **24,828 intervals (69.6%)** represent sensor-only inter-survey periods.

2. **Digital Biomarker: Physical Activity Reduction:**
   - Daily active step count (`steps_awake_mean`) exhibits a strong, monotonic decline as depression severity increases.
   - Non-depressed subjects (PHQ-9 0–4) log an average of **9,463 steps/day**, whereas severely depressed subjects (PHQ-9 20–27) average **6,980 steps/day**—representing a **26.2% drop in daily physical movement**.
   - Active day counts decrease while sedentary day counts increase significantly ($r = +0.229$).

3. **Digital Biomarker: Sleep Fragmentation & Efficiency:**
   - **Sleep Efficiency Drop:** Weekday sleep efficiency (`sleep_ratio_asleep_in_bed_weekday_mean`) falls linearly from **90.9%** in minimal depression down to **84.8%** in severe depression.
   - **In-Bed Latency:** While total weekday sleep duration remains ~6 hours across all categories, severely depressed individuals spend **34 minutes longer in bed** per night without sleeping.
   - **Erratic Sleep Schedules:** Sleep duration variability (IQR) increases by **31.2%** in severe depression (96.2 min vs 126.1 min).
   - **Circadian Shift:** Median bedtime delays by **~45 minutes** in severe depression (10:44 PM vs 11:27 PM).

4. **Clinical & Socioeconomic Correlates:**
   - Top clinical positive predictors of PHQ-9 score: **Comorbid Migraines** ($r = +0.353$), **Financial Strain** ($r = +0.351$), **Neuropathic Pain** ($r = +0.335$), **Migraine Days/Month** ($r = +0.291$), and **Arthritis** ($r = +0.223$).
   - Participants with chronic neuropathic pain exhibit a mean PHQ-9 score of **12.11** compared to **5.83** in non-affected participants.

---

## 1. Dataset Dimensions & Schema Overview

| Attribute | Metric |
| :--- | :--- |
| **Total Rows** | 35,694 |
| **Total Columns** | 154 |
| **Unique Participants** | 4,948 |
| **Period Range per Subject** | 1 to 12 periods (Mean: 7.21, Median: 8) |
| **Index Structure** | `user_id` + `period_num` (e.g. `4938_1`) |
| **Labeled PHQ-9 Rows** | 10,866 (30.4%) |
| **Unlabeled Sensor Rows** | 24,828 (69.6%) |

### Feature Categorization (154 Columns)

- **Actigraphy / Physical Activity (42 features):** `steps_awake_mean`, `steps_mvpa_sum_recent`, `steps_lpa_sum_recent`, `steps__active_day_count_`, `steps__sedentary_day_count_`, `steps__rolling_6_sum__max__score`, `steps__streaks__countDistinct__score`, linear trend scores/coeffs.
- **Sleep & Circadian Rhythms (48 features):** `sleep_asleep_weekday_mean`, `sleep_asleep_weekend_mean`, `sleep_in_bed_weekday_mean`, `sleep_in_bed_weekend_mean`, `sleep_ratio_asleep_in_bed_weekday_mean`, `sleep_asleep_iqr`, `sleep_main_start_hour_adj_median`, `sleep__hypersomnia_count_`, `sleep__hyposomnia_count_`.
- **Demographics & Physical Status (18 features):** `sex`, `birthyear` (age range 1940–2000), `educ`, `height`, `weight`, `bmi`, `race_white`, `race_black`, `race_hispanic`, `race_asian`, `race_other`, `pregnant`, `household`, `insurance`, `money`, `money_assistance`.
- **Clinical Comorbidities & Interventions (22 features):** `comorbid_migraines`, `num_migraine_days`, `meds_migraine`, `comorbid_neuropathic`, `comorbid_arthritis`, `comorbid_cancer`, `comorbid_diabetes_typ1`, `comorbid_diabetes_typ2`, `comorbid_gout`, `comorbid_ms`, `comorbid_osteoporosis`, `med_start`, `med_stop`, `life_meditation`, `life_stress`.
- **Target Depression Variables (4 features):** `phq9_score_start`, `phq9_score_end`, `phq9_cat_start`, `phq9_cat_end`.

---

## 2. Missing Value Profiling & Data Completeness

Missingness was evaluated across all 154 features and categorized into completeness tiers:

| Tier | Missingness Range | Feature Count | Examples |
| :--- | :--- | :--- | :--- |
| **Tier 1: Complete** | 0% missing | 41 features | `sex`, `birthyear`, `comorbid_migraines`, `steps__sedentary_day_count_` |
| **Tier 2: High Quality** | < 5% missing | 39 features | `steps_awake_mean`, `bmi`, `sleep_asleep_weekday_mean`, `educ` |
| **Tier 3: Moderate** | 5% – 30% missing | 44 features | `steps__mvpa__sum__score`, `sleep__nap_count__score` |
| **Tier 4: Sparse** | 30% – 70% missing | 28 features | `phq9_score_start` (69.6%), `sleep_asleep_iqr` (37.7%) |
| **Tier 5: Very Sparse** | ≥ 70% missing | 4 features | `steps_mvpa_iqr` (86.3%), `steps__mvpa__sum__score_` (75.8%) |

---

## 3. Target Variable Analysis: PHQ-9 Severity Distributions

The primary target is `phq9_score_start` (range 0 to 27) and its categorical mapping `phq9_cat_start`:

| Category ID | Category Name | Score Range | Count | Labeled % | Total Dataset % |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | None / Minimal | 0 – 4 | 4,024 | 37.0% | 11.3% |
| **1** | Mild | 5 – 9 | 3,267 | 30.1% | 9.2% |
| **2** | Moderate | 10 – 14 | 1,980 | 18.2% | 5.5% |
| **3** | Moderately Severe | 15 – 19 | 1,054 | 9.7% | 3.0% |
| **4** | Severe | 20 – 27 | 541 | 5.0% | 1.5% |
| **NaN** | Unlabeled Interval | N/A | 24,828 | — | 69.6% |

### Score Distribution Metrics
- **Mean:** 7.66
- **Std Dev:** ±6.03
- **Median (50%):** 6.00
- **25th Percentile:** 3.00
- **75th Percentile:** 11.00

---

## 4. Digital Biomarker Analysis: Actigraphy & Sleep Dynamics

Comparing actigraphy and sleep features across PHQ-9 severity categories highlights distinct digital signatures:

| Feature Metric | Minimal (0) | Mild (1) | Moderate (2) | Mod. Severe (3) | Severe (4) | Trend Direction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Steps Awake Mean** | 9,463 | 8,349 | 7,799 | 7,191 | 6,980 | 📉 Monotonic Decrease (-26.2%) |
| **Weekday Sleep Efficiency (%)** | 90.9% | 89.1% | 86.8% | 85.5% | 84.8% | 📉 Monotonic Decrease |
| **Weekday In-Bed Minutes** | 397.4 m | 414.4 m | 427.1 m | 426.3 m | 431.1 m | 📈 Monotonic Increase (+33.7 min) |
| **Sleep Duration IQR (Mins)** | 96.2 m | 102.1 m | 109.3 m | 117.9 m | 126.1 m | 📈 Increased Variability (+31.2%) |
| **Hypersomnia Episodes Count** | 0.21 | 0.32 | 0.40 | 0.44 | 0.47 | 📈 More than doubled (+123.8%) |
| **Hyposomnia Episodes Count** | 1.46 | 1.58 | 1.70 | 1.83 | 1.90 | 📈 Steady Increase (+30.1%) |
| **Median Bedtime (Adjusted)** | 22.73 (10:44 PM) | 22.82 | 23.08 | 23.27 | 23.45 (11:27 PM) | ⏰ Delayed Bedtime (+41 mins) |

---

## 5. Statistical Correlation Analysis

### Top 10 Positive Correlations with PHQ-9 Score
1. `comorbid_migraines` ($r = +0.353$)
2. `money` (financial strain level) ($r = +0.351$)
3. `comorbid_neuropathic` ($r = +0.335$)
4. `num_migraine_days` ($r = +0.291$)
5. `steps__sedentary_day_count_` ($r = +0.229$)
6. `money_assistance` ($r = +0.227$)
7. `comorbid_arthritis` ($r = +0.223$)
8. `meds_migraine` ($r = +0.215$)
9. `sleep_ratio_asleep_in_bed_iqr` ($r = +0.173$)
10. `bmi` ($r = +0.161$)

### Top 10 Negative Correlations with PHQ-9 Score
1. `sex` ($r = -0.223$)
2. `steps_rolling_6_median_recent` ($r = -0.222$)
3. `steps__rolling_6_sum__max__intercept` ($r = -0.217$)
4. `steps_rolling_6_max_recent` ($r = -0.207$)
5. `steps_awake_mean` ($r = -0.203$)
6. `steps__awake__sum__intercept` ($r = -0.184$)
7. `steps__active_day_count_` ($r = -0.182$)
8. `med_nonmed_dnu` ($r = -0.176$)
9. `steps_lpa_sum_recent` ($r = -0.167$)
10. `sleep_ratio_asleep_in_bed_weekday_mean` ($r = -0.159$)

---

## 6. Longitudinal PHQ-9 Category Transition Matrix

Evaluating category transitions from interval start (`phq9_cat_start`) to interval end (`phq9_cat_end`) demonstrates high baseline stability with gradual migration:

| Start Category \ End Category | 0: Minimal | 1: Mild | 2: Moderate | 3: Mod. Severe | 4: Severe |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0: None / Minimal** | **77.6%** | 18.6% | 3.0% | 0.5% | 0.2% |
| **1: Mild** | 27.1% | **48.6%** | 18.8% | 4.5% | 1.0% |
| **2: Moderate** | 7.7% | 34.3% | **38.4%** | 15.3% | 4.3% |
| **3: Moderately Severe** | 3.0% | 16.2% | 32.2% | **32.4%** | 16.1% |
| **4: Severe** | 1.3% | 6.5% | 19.6% | 31.2% | **41.4%** |

---

## 7. Machine Learning Modeling Recommendations

### 1. Cross-Validation & Data Splitting Strategy
- **CRITICAL:** Use `GroupKFold(groups=user_id)` or `GroupShuffleSplit`.
- Because participants contribute an average of 7.2 periods, random row-wise splitting will cause extreme participant data leakage and result in overly optimistic, invalid evaluation metrics.

### 2. Handling Unlabeled Data (69.6%)
- **Supervised Training:** Train baseline supervised classifiers (LightGBM, XGBoost, CatBoost) on the 10,866 labeled rows.
- **Semi-Supervised / Self-Training:** Utilize pseudo-labeling or self-supervised contrastive learning on the 24,828 unlabeled sensor rows to pre-train representation encoders.

### 3. Class Imbalance Mitigation
- Severe (5.0%) and Moderately Severe (9.7%) form minority classes.
- Implement class weighting (`class_weight='balanced'`), Focal Loss, or SMOTE oversampling in training pipelines.

### 4. High-Value Feature Engineering Ideas
- **Restlessness Ratio:** `steps_awake_mean / sleep_in_bed_weekday_mean`
- **Sleep Debt Indicator:** `sleep_asleep_weekend_mean - sleep_asleep_weekday_mean` (Social Jetlag)
- **Clinical Burden Score:** Composite index combining `comorbid_migraines`, `comorbid_neuropathic`, and `comorbid_arthritis`.

---

## Artifact Files Created
- **Jupyter Notebook:** [Depression_Detection_EDA.ipynb](file:///d:/projects/DepDet/Depression_Detection_EDA.ipynb) (Fully executed with code & plots)
- **HTML Dashboard:** [EDA_Report.html](file:///d:/projects/DepDet/EDA_Report.html) (Self-contained interactive dashboard)
- **Markdown Report:** [EDA_Report.md](file:///d:/projects/DepDet/EDA_Report.md) (This comprehensive document)
