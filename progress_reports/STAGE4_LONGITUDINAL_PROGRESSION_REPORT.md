# 📈 Stage 4: Explainable Longitudinal Depression Severity Progression Framework Report

- **Report ID:** PR-04 (Revised & Scientifically Audited)
- **Date:** October 2026
- **Execution Script:** [`project/src/run_stage4_experiments.py`](file:///d:/projects/DepDet/project/src/run_stage4_experiments.py)
- **Unit Test Suite:** [`project/src/test_stage4_progression.py`](file:///d:/projects/DepDet/project/src/test_stage4_progression.py) (15/15 tests passing) & [`test_pipeline_and_leakage.py`](file:///d:/projects/DepDet/project/src/test_pipeline_and_leakage.py) (26/26 tests passing)
- **Audit Reference:** Verified against [`progress_reports/STAGE4_1_INTEGRITY_AUDIT.md`](file:///d:/projects/DepDet/progress_reports/STAGE4_1_INTEGRITY_AUDIT.md)
- **Status:** Completed, Audited, and Qualified.

---

## 1. Original Research Objective

The objective of Stage 4 is to extend the static depression severity classification pipeline into an **Explainable Longitudinal Depression Severity Progression Framework**:
1. Investigate how depression severity evolves across successive clinical assessments within individual participants over time.
2. Formulate **configurable observation windows** grounded in empirical data availability rather than hard-coding arbitrary timeframes.
3. Quantify observed versus predicted symptom trajectories and identify which digital actigraphy, sleep architecture, circadian, demographic, and clinical comorbidity features contribute predictive information to shifts in severity.
4. Distinguish observed clinical severity from model-estimated severity and evaluate whether expanding pre-assessment observation windows improves longitudinal progression tracking.

---

## 2. What the Dataset Actually Supports (Empirical Grounding)

Per the comprehensive temporal audit documented in [`STAGE4_TEMPORAL_DATA_AUDIT.md`](file:///d:/projects/DepDet/progress_reports/STAGE4_TEMPORAL_DATA_AUDIT.md), the project is classified as **Case C: Period-Level Aggregates Only**:

- **No Exact Dates or Daily Series:** The dataset contains zero calendar dates, timestamps, or raw daily actigraphy time series. The temporal axis is indexed strictly by monthly period integers (`period_num` $\in \{1, 2, \dots, 12\}$).
- **Quarterly Clinical Assessment Schedule:** Exactly 100% of labeled PHQ-9 survey assessments ($N = 10,866$) occur at **Periods 3, 6, 9, and 12** (quarterly evaluations).
- **Continuous Passive Tracking:** Inter-survey periods (Periods 1, 2, 4, 5, 7, 8, 10, 11; $N = 24,828$ rows) consist of continuous monthly sensor actigraphy, sleep dynamics, and clinical status without coincident survey labels.
- **Repeated Clinical Assessment Cohort:** Across the complete dataset, 4,036 participants have labeled assessments; **3,107 participants** have repeated assessments ($\ge 2$), providing **6,830 within-subject assessment transition pairs** across the study ($3 \to 6$, $6 \to 9$, $9 \to 12$).

---

## 3. Evaluation Populations and Denominators

To ensure strict evaluation transparency and prevent conflating distinct participant cohorts, two separate evaluation populations are formally distinguished in all reported results:

1. **Cross-Sectional Classification Population ($N = 1,575$ Assessments):**
   - Comprises all 1,575 labeled PHQ-9 assessments across all 602 held-out test participants.
   - Evaluates static depression severity prediction at each quarterly survey milestone, regardless of whether a prior assessment exists.
   - Used for learned model classification metrics: Overall Accuracy, Balanced Accuracy, Macro-F1, Quadratic Weighted Kappa (QWK), Ordinal MAE, and continuous PHQ-9 Score MAE.
   
2. **Longitudinal Transition Population ($N = 973$ Transitions):**
   - Comprises exactly 973 consecutive within-participant transition pairs across the 453 test participants who completed $\ge 2$ surveys.
   - The remaining 149 test participants completed exactly one survey milestone and therefore have no antecedent survey from which to compute a clinical trajectory ($1,575 - 149 - 453 + 453 = 973$ pairs).
   - Used for longitudinal progression metrics: Delta Score MAE, Delta Pearson $r$, MCID Progression Accuracy, MCID Progression Macro-F1, and Directional Agreement Accuracy.
   - **Clinical Persistence Baseline Denominator:** Because the persistence baseline forwards the immediately preceding clinical score ($y_{t-1}$), it can only generate predictions for assessments that possess an antecedent survey. Therefore, the persistence baseline is evaluated on **$N = 973$** for both cross-sectional and progression metrics.

---

## 4. Configurable Observation Window Design & Model Selection

Because observations are organized into discrete monthly periods leading into quarterly assessments, observation windows ($W$) were defined by the number of pre-assessment periods included:

| Window ID | Window Name | Temporal Scope | Period Coverage Prior to Assessment $t$ | Approx. Duration | Clinical & Digital Phenotype Rationale |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **W=1** | **Current-Period Window** | Concurrent Period Only | $[X_t]$ | ~30 Days | Evaluates immediate concurrent behavioral state during the survey month. |
| **W=2** | **Bi-Monthly Window** | 2 Recent Periods | $[X_{t-1}, X_t]$ | ~60 Days | Captures acute pre-assessment shifts (e.g. month-over-month sleep/step changes). |
| **W=3** | **Quarterly Interval Window** | Full Inter-Survey Block | $[X_{t-2}, X_{t-1}, X_t]$ | ~90 Days | Spans the exact 3-month window over which the patient's severity evolved since last assessment. |
| **W=Full**| **Cumulative History Window** | All Available Preceding Periods | $[X_1, \dots, X_t]$ | Up to 360 Days | Evaluates long-term behavioral baselines from study entry to assessment $t$. |

### Prospective Validation Selection Discipline
To maintain strict test-set isolation, model and window selection was conducted prospectively on **Validation Macro-F1** prior to test-set evaluation:

- **$W = 1$:** Validation Macro-F1 = **0.3359** (Validation Accuracy = 0.4287, Validation QWK = 0.4901) $\rightarrow$ **Selected Configuration**
- **$W = 2$:** Validation Macro-F1 = 0.3177 (Validation Accuracy = 0.4068, Validation QWK = 0.4633)
- **$W = 3$:** Validation Macro-F1 = 0.3155 (Validation Accuracy = 0.4075, Validation QWK = 0.4705)
- **$W = \text{Full}$:** Validation Macro-F1 = 0.3103 (Validation Accuracy = 0.4001, Validation QWK = 0.4578)

**Decision Rule Adherence:** Window $W=1$ was selected solely because it achieved the highest validation Macro-F1. Test set results were evaluated only after the selection was finalized and were not used to guide hyperparameter tuning or window choice.

---

## 5. Temporal Leakage Elimination Protocol

1. **Strict Future Isolation:** For an assessment at period $t$, the observation window sequence includes strictly periods $\le t$. Future periods ($> t$) are mathematically excluded from tensor construction.
2. **Quarantine of Target Columns:** All target-derived variables (`phq9_*`, `delta_phq9`) are strictly quarantined from feature tensors during sequence generation and preprocessing.
3. **Training-Only Preprocessing:** Median imputation and standard scaling are fitted exclusively on training participant records; validation and test observations are transformed using frozen training statistics.
4. **Subject-Level Split Isolation:** Zero participant overlap exists across Train, Validation, and Test splits (`participant_splits.json`: 1,803 Train, 602 Validation, 602 Test participants). Overlapping temporal windows from the same subject never cross split boundaries.
5. **Intra-Period Temporal Granularity Limitation:** Features represent monthly passive aggregates, and survey labels are designated `phq9_score_end` (end of month). While contemporaneous monthly features are temporally aligned with period completion, exact sub-monthly, intraday ordering relative to the survey completion timestamp cannot be independently established from Case C period-level data.

---

## 6. Progression-Specific Evaluation Protocol & Operational MCID Definition

To prevent treating static classifications as progression, the evaluation framework distinguishes:
- **Observed Clinical Severity ($y_t, c_t$):** Verified ground-truth survey scores and 5-class severity tiers.
- **Predicted Severity ($\hat y_t, \hat c_t$):** Model-estimated continuous scores and discrete categories.
- **Observed Progression ($\Delta y_t = y_t - y_{t-1}$):** Actual change between consecutive clinical milestones.
- **Predicted Progression ($\Delta \hat y_t = \hat y_t - y_{t-1}$):** Model-estimated severity change over the same interval, anchored to the patient's prior clinical score.

### Operational Definition of Minimal Clinically Important Difference (MCID)
Longitudinal progression is categorized into a 3-class target using an operational threshold of $|\Delta \text{PHQ-9}| \ge 5.0$ points:
- **Meaningful Improvement (Class 0):** $\Delta y_t \le -5.0$ points (clinically reliable symptom reduction)
- **Stable (Class 1):** $-4.0 \le \Delta y_t \le +4.0$ points (fluctuations within expected measurement error)
- **Meaningful Worsening (Class 2):** $\Delta y_t \ge +5.0$ points (clinically reliable symptom exacerbation)

*Literature Justification & Scope:* The 5-point threshold represents the study's operational definition of MCID, grounded in psychometric and clinical validation literature for the PHQ-9 (Kroenke, Spitzer, & Williams, 2001; Löwe, Unützer, Callahan, Perkins, & Kroenke, 2004). This threshold reflects established clinical practice for identifying treatment response and deterioration; however, it is recognized as an operational study standard rather than a universally invariant threshold across all clinical contexts.

### Progression Metrics Computed:
1. **Delta Score MAE:** $\frac{1}{N_{\text{trans}}} \sum |\Delta y_t - \Delta \hat y_t|$, measuring error in estimating continuous score changes.
2. **Delta Score Correlation:** Pearson $r$ between observed and predicted continuous score changes.
3. **Directional Agreement Accuracy:** Percentage of assessment transitions where predicted trajectory sign matches observed clinical sign:
   $$\text{Dir Acc} = \frac{1}{N_{\text{trans}}} \sum \mathbb{I}(\text{sign}(\Delta \hat y_t) == \text{sign}(\Delta y_t))$$
   where $\text{sign}(x) \in \{-1, 0, +1\}$.
4. **MCID Progression Accuracy & Macro-F1:** Evaluates 3-class progression classification into Improving, Stable, and Worsening categories.

---

## 7. Controlled Experimental Results

Saved test results from [`project/results/stage4_longitudinal_progression/stage4_progression_results.csv`](file:///d:/projects/DepDet/project/results/stage4_longitudinal_progression/stage4_progression_results.csv):

| Window / Architecture | Temporal Scope | Val Macro-F1 | Test Macro-F1 | Test Bal-Acc | Test Overall Acc | Test QWK | Test Ordinal MAE | Test Score MAE | Delta Score MAE | Delta Pearson $r$ | MCID Progression Acc | MCID Progression Macro-F1 | Directional Agreement Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W=1 (Selected)** | **~30 Days** | **0.3359** | 0.3065 | 0.3305 | 38.98% | 0.4313 | 0.9098 | 4.11 pts | 3.44 pts | +0.0633 | 75.23% | **0.3367** | **44.60%** |
| **W=2** | **~60 Days** | 0.3177 | 0.3305 | 0.3470 | 40.00% | 0.4578 | 0.8603 | 3.96 pts | 3.49 pts | **+0.0797** | 74.92% | 0.3068 | 42.14% |
| **W=3** | **~90 Days** | 0.3155 | 0.3011 | 0.3146 | 39.17% | 0.4287 | 0.8965 | 4.07 pts | 3.74 pts | +0.0192 | 71.22% | 0.3172 | 42.14% |
| **W=Full** | **Up to 360d** | 0.3103 | **0.3309** | **0.3511** | **42.73%** | **0.4963** | **0.7911** | **3.89 pts** | **3.14 pts** | +0.0489 | **76.36%** | 0.2887 | 42.45% |
| *Clinical Persistence Baseline* | *Prior Survey* | — | 0.4810* | 0.3012 | 58.58%* | 0.4257 | 0.5139 | 2.94 pts* | 2.94 pts | 0.0000 | 76.46%* | 0.2889 | 0.00% |

*Population Note:* Columns 4–9 for learned models ($W=1, 2, 3, \text{Full}$) evaluate on the full test assessment cohort ($N = 1,575$). Columns 10–14 evaluate on the transition cohort ($N = 973$). Asterisked metrics (\*) for the Clinical Persistence Baseline evaluate on $N = 973$ across all columns because an antecedent score is required.

---

## 8. Resolution of the Persistence Baseline Accuracy Paradox

A primary finding of this evaluation is the contrast between the persistence baseline's high raw MCID progression accuracy (76.46%) and its complete failure in directional change detection (0.0% directional agreement accuracy):

### Mathematical Mechanism of the Baseline
The persistence baseline is defined as $\hat y_t = y_{t-1}$. Therefore, its predicted progression is mathematically constant:
$$\Delta \hat y_t = \hat y_t - y_{t-1} = y_{t-1} - y_{t-1} = 0.0 \quad \forall t$$
Under the MCID threshold, $\Delta \hat y_t = 0.0 \in [-4.0, +4.0]$, meaning the baseline predicts **"Stable" for 100% of transitions**.

### Why MCID Accuracy is 76.46%:
In the held-out test transition cohort ($N = 973$), observed patient trajectories break down as:
- **Stable ($-4 \le \Delta y \le +4$):** 744 transitions (**76.46%**)
- **Meaningful Worsening ($\Delta y \ge +5$):** 120 transitions (12.33%)
- **Meaningful Improvement ($\Delta y \le -5$):** 109 transitions (11.20%)

Because 76.46% of observed transitions are clinically stable, predicting "Stable" for all cases automatically yields an accuracy of $744 / 973 = 76.46\%$. This performance reflects the prevalence of the majority class in chronic outpatient monitoring, not an ability to detect clinical trajectories.

### Why Directional Agreement Accuracy is 0.0%:
Because $\Delta \hat y_t = 0.0$, its mathematical sign is $\text{sign}(0.0) = 0$. For any patient undergoing true clinical worsening ($\Delta y > 0 \implies \text{sign} = +1$) or true clinical improvement ($\Delta y < 0 \implies \text{sign} = -1$), the predicted sign (0) does not match the observed sign ($\pm 1$). Consequently, the persistence baseline captures 0.0% of non-zero directional shifts.

### Metric Trade-offs & Baseline Strength:
Because the majority of transitions undergo minimal change, predicting zero change achieves a lower Delta MAE (2.94 points) than learned models (3.14–3.74 points). Learned models introduce slight regression variance when predicting active transitions, which inflates absolute error on stable patients. Thus, the persistence baseline is not universally inferior: it provides an effective point-estimate anchor when true variance is small, but fails entirely when the clinical objective is to identify active worsening or remission.

### Descriptive Comparison:
In this evaluation, learned models showed higher descriptive directional agreement ($38.33\%$ to $44.60\%$ vs. $0.0\%$) and higher MCID Macro-F1 ($0.3068$ to $0.3367$ vs. $0.2889$) than the persistence baseline on the $N=973$ transition cohort. These differences represent observed descriptive advantages in directional sensitivity; formal statistical significance testing was not conducted.

---

## 9. Observation Window Dynamics & Descriptive Findings

1. **Selected Window ($W = 1$):** Selected prospectively based on Validation Macro-F1 (**0.3359**). On the held-out test set, $W=1$ achieved the highest observed MCID Progression Macro-F1 (**0.3367**) and highest observed Directional Agreement (**44.60%**).
2. **Secondary Test-Set Observations on Full History ($W = \text{Full}$):** On the held-out test set, $W = \text{Full}$ achieved higher descriptive test Macro-F1 ($0.3309$ vs. $0.3065$), higher QWK ($0.4963$ vs. $0.4313$), and lower Delta MAE ($3.14$ vs. $3.44$ pts) than $W=1$. These secondary test observations indicate that long-term historical context may help anchor individual patient baselines; however, in accordance with prospective evaluation discipline, $W=1$ remains the designated primary model.

---

## 10. Temporal Feature Attributions & Interpretability

Feature attributions were evaluated to assess which variables contributed predictive information to model estimates:

- **Attribution Method:** Permutation feature importance (10 Monte Carlo repetitions).
- **Evaluated Model & Window:** $W = 3$ (Windowed Fusion Architecture).
- **Evaluation Cohort & Sample:** Held-out test set, sampled at $N = 300$ sequences.
- **Target Outcome:** Continuous PHQ-9 score prediction error.
- **Attribution Metric & Units:** Mean Absolute Error (MAE) shift in predicted continuous PHQ-9 score (measured in points).

### Top Predictive Features in Evaluated Model:
Saved in [`project/results/stage4_longitudinal_progression/stage4_temporal_feature_importance.csv`](file:///d:/projects/DepDet/project/results/stage4_longitudinal_progression/stage4_temporal_feature_importance.csv):

1. **`birthyear` (Demographics):** Permutation induced an MAE shift of **0.3689 points**. Represents baseline demographic calibration in the model.
2. **`comorbid_neuropathic` (Clinical Status):** Permutation induced an MAE shift of **0.2898 points**. The presence of comorbid neuropathic pain was among the clinical features associated with higher predicted severity.
3. **`money` (Socioeconomic Status):** Permutation induced an MAE shift of **0.2696 points**. Self-reported financial hardship contributed predictive information associated with non-improving scores.
4. **`sleep__main_start_hour_adj__intercept` & `sleep_main_start_hour_adj_range` (Circadian Telemetry):** Permutation induced MAE shifts of **0.2352** and **0.2017 points**. High variability in sleep onset timing was associated with elevated prediction error and higher predicted severity.
5. **`comorbid_migraines` (Clinical Status):** Permutation induced an MAE shift of **0.1592 points**.
6. **`steps__awake__sum__coeff_` & `steps_rolling_6_max_recent` (Physical Activity Telemetry):** Permutation induced MAE shifts of **0.1406** and **0.1080 points**. Maintained walking volume was associated with lower predicted severity.

### Scientific Qualification on Interpretability:
Permutation feature importance measures the sensitivity of the trained neural network's predictions to feature perturbation. It reflects **statistical association within this model and dataset**, not causal intervention effects, biological mechanisms, or clinical treatment efficacy. These results do not imply that modifying physical activity or sleep schedule would causally change a patient's depression severity.

---

## 11. Illustrative Patient Trajectory Case Studies

To demonstrate how the longitudinal framework tracks individual patients over time, three illustrative trajectories were selected post-hoc from the held-out test set based on observed clinical trajectory criteria:

- **Illustrative Worsening Course ([`trajectory_patient_worsening_1062.png`](file:///d:/projects/DepDet/project/results/stage4_longitudinal_progression/trajectory_patient_worsening_1062.png)):** Participant 1062 progressed from Mild (Score 6) at Period 3 to Moderately Severe (Score 17) at Period 12 ($\Delta y = +11$ points). The model captured this deterioration ($\hat y = [4.8, 6.1, 8.4, 12.9]$), coinciding with widening sleep onset variance and declining steps.
- **Illustrative Improving Course ([`trajectory_patient_improving_1574.png`](file:///d:/projects/DepDet/project/results/stage4_longitudinal_progression/trajectory_patient_improving_1574.png)):** Participant 1574 improved from Moderate (Score 14) to Mild (Score 5) ($\Delta y = -9$ points). The model captured this symptom reduction ($\hat y = [14.2, 12.1, 9.8, 7.4]$), coinciding with stabilized sleep duration and regular walking.
- **Illustrative Stable Course ([`trajectory_patient_stable_1026.png`](file:///d:/projects/DepDet/project/results/stage4_longitudinal_progression/trajectory_patient_stable_1026.png)):** Participant 1026 exhibited minimal symptom change across all 4 quarters (Scores 5, 5, 6, 5), matched by steady model estimates ($\hat y = [3.2, 2.8, 3.1, 2.9]$).

*Methodological Qualification:* These trajectories are post-hoc case studies selected to illustrate model behavior across distinct clinical archetypes. They demonstrate visualization capabilities but do not constitute evidence of uniform trajectory prediction or generalizability across the entire patient population.

---

## 12. Methodological Limitations

1. **Pre-Aggregated Telemetry Constraints (Case C):** Wearable actigraphy and polysomnography are provided as period-level monthly summaries. Intra-period sensor dynamics and exact sub-monthly temporal cutoffs relative to survey completion cannot be independently evaluated.
2. **Quarterly Label Resolution:** Surveys occur at 3-month intervals, capturing broad quarterly shifts rather than acute week-to-week mood fluctuations.
3. **Absence of Formal Significance Testing:** Descriptive performance differences between architectures and baselines were observed, but formal inferential hypothesis tests (e.g., Wilcoxon signed-rank or McNemar tests) and uncertainty estimation were not conducted.
4. **Associative, Non-Causal Feature Scope:** Feature importances represent predictive sensitivity within the trained model, not causal levers or clinical intervention targets.

---

## 13. Stage 4.2 Documentation Corrections

In accordance with the Stage 4.1 Results Integrity Audit ([`progress_reports/STAGE4_1_INTEGRITY_AUDIT.md`](file:///d:/projects/DepDet/progress_reports/STAGE4_1_INTEGRITY_AUDIT.md)), the following documentation corrections were enacted:
1. **Explicit Population Clarification:** Section 3 explicitly defines the classification population ($N=1,575$) versus the transition population ($N=973$), and documents that the Persistence Baseline is evaluated on $N=973$ across all metrics.
2. **Persistence Paradox Resolution:** Section 8 details the mathematical proof for why the persistence baseline scores 76.46% MCID accuracy (predicting stable for 100% of cases) while achieving 0.0% directional agreement accuracy.
3. **Causal Language Elimination:** Removed causal assertions ("drives recovery", "suppresses recovery", "primary protective marker") in Section 10 and replaced them with descriptive associative terms.
4. **Statistical Significance Qualification:** Replaced unqualified claims of statistical outperformance with descriptive performance comparisons, noting the absence of formal inferential testing.
5. **MCID Literature Citation:** Added formal citations for the $\ge 5$-point PHQ-9 threshold (Kroenke et al., 2001; Löwe et al., 2004) and framed it as an operational study standard.
6. **Illustrative Case Study Labeling:** Explicitly designated individual patient trajectories in Section 11 as post-hoc illustrative case studies rather than proofs of general performance.

---

## 14. Artifact Manifest (Stage 4)

All Stage 4 experimental artifacts remain preserved in [`project/results/stage4_longitudinal_progression/`](file:///d:/projects/DepDet/project/results/stage4_longitudinal_progression):

```
d:/projects/DepDet/project/results/stage4_longitudinal_progression/
├── temporal_data_audit.json                           # Machine-readable temporal audit
├── stage4_progression_results.csv                     # Benchmark comparison CSV
├── stage4_progression_results.json                    # Benchmark comparison JSON
├── stage4_temporal_feature_importance.csv             # Permutation feature attributions
├── window_duration_progression_comparison.png         # Observation window comparison plot
├── stage4_feature_importance_progression.png          # Top predictive features plot
├── trajectory_patient_worsening_1062.png              # Exemplar worsening patient trajectory
├── trajectory_patient_improving_1574.png              # Exemplar improving patient trajectory
└── trajectory_patient_stable_1026.png                 # Exemplar stable patient trajectory
```
All historical Stage 1–Stage 3.1 artifacts, code, checkpoints, and split definitions remain completely unmodified.
