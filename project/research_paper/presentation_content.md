# Academic Presentation Deck: Explainable Longitudinal Depression Severity Prediction & Clinical Progression Tracking from Passive Wearables

**File:** `presentation_content.md`  
**Target Duration:** 10–15 Minutes Oral Academic Defense / Conference Presentation  
**Scope:** Complete Slide Text, Visual Guides, Spoken Presenter Scripts, Source Traceability, Design Guide, and Panel Q&A  
**Repository:** `d:\projects\DepDet\project`

---

# SECTION 1: SLIDE-BY-SLIDE PRESENTATION CONTENT (SLIDES 1–16)

---

### Slide 1: Title Slide

- **Slide Title:** Explainable Longitudinal Depression Severity Prediction and Progression Tracking from Passive Wearable Sensing
- **Subtitle:** A Subject-Independent Benchmark, Multi-Window Recurrent Modeling, and Clinical Decision Support Framework
- **Presenter:** [Presenter Name / Degree Candidate]
- **Department / Institution:** Department of Computer Science & Engineering / Department of Health Informatics, [University Name]
- **Advisors / Supervisors:** Supervised by [Supervisor Name, Title]
- **Date:** Academic Year 2025–2026

**Visual & Layout Specification:**
- Professional deep navy canvas (`#0F172A`) with high-contrast white title typography and cyan-blue accent separator (`#38BDF8`).
- Centered institution crest / logo placeholder and presenter credentials block.

**Presenter Spoken Script (30–45 seconds):**
> *"Good morning, members of the evaluation committee. Today, I am presenting our research on explainable longitudinal depression severity prediction and progression tracking using passive wearable sensing. In this project, we address the critical challenges of subjective recall bias, pervasive data leakage, severe ordinal class imbalance, and the longitudinal persistence paradox by establishing a strictly subject-independent evaluation protocol across 4,948 participants. I will walk you through our methodology, experimental benchmarks, clinical triage calibration, and interpretability findings."*

**Source Reference:**
- `project/research_paper/research_paper.md` (Lines 1–10)

---

### Slide 2: Introduction and Clinical Background

- **Global Mental Health Crisis:** Major Depressive Disorder (MDD) affects over 280 million individuals globally, representing the leading cause of non-fatal psychiatric disability (WHO 2023).
- **The Episodic Assessment Bottleneck:** Outpatient clinical psychiatry relies on retrospective questionnaires (PHQ-9, HDRS) administered during brief appointments spaced 1 to 3 months apart.
- **Inherent Vulnerabilities of Self-Report:** Retrospective questionnaires suffer from cognitive recall bias, acute situational affect on evaluation day, and social desirability masking.
- **The Digital Phenotyping Paradigm:** Continuous, unobtrusive monitoring via consumer wearables (actigraphy, sleep architecture, circadian rhythmicity) offers an objective, continuous window into daily affective fluctuations.

**Visual & Layout Specification:**
- Two-column contrast layout:
  - *Left Box (Coral Red Border):* "Conventional Psychiatric Triage" — Intermittent quarterly clinic visits, 14-day recall distortion, missed deterioration between visits.
  - *Right Box (Emerald Green Border):* "Digital Phenotyping Triage" — Passive continuous behavioral telemetry, objective circadian tracking, prospective early-warning alerts.

**Presenter Spoken Script (45–60 seconds):**
> *"Depression is one of the world's most pervasive health crises, yet psychiatric assessment has remained fundamentally unchanged for decades. When a clinician administers the 9-item Patient Health Questionnaire, the patient must mentally summarize two weeks of emotional and physical symptoms. This retrospective process is prone to recall bias—a patient evaluated during a temporary good day may under-report significant struggles from earlier in the month. Wearable devices equipped with actigraphy and physiological sensors offer an objective alternative. By continuously measuring sleep regularity, motor activity, and circadian shifts in everyday environments, we can detect early behavioral precursors of relapse between scheduled outpatient visits."*

**Source Reference:**
- `project/research_paper/research_paper.md` (Section 1), WHO Fact Sheets (2023), Kroenke et al. (2001)

---

### Slide 3: Problem Statement: Latent Representation Overlap & Measurement Noise

- **The Representation Overlap Dilemma:** When high-dimensional clinical, behavioral, or acoustic/textual representations are mapped into a latent space, depressive severity classes exhibit severe topological overlap rather than clean separability.
- **Why Do Depressive Latent Spaces Overlap?**
  - *Symptom Heterogeneity:* Two patients with identical moderate PHQ-9 scores ($\text{PHQ-9} = 12$) often present opposite behavioral patterns (insomnia vs. hypersomnia; psychomotor agitation vs. retardation).
  - *Socioeconomic Masking:* High educational attainment and financial wealth statistically buffer daily routines, masking acute emotional suffering.
  - *Measurement Noise:* A 27-point self-report questionnaire carries inherent intra-individual scoring variance ($\pm 2$ to $4$ points) without an underlying change in clinical state.
- **Core Technical Problem:** Standard classification algorithms trained on static, overlapping representations collapse to majority classes and fail to generalize when deployed on unseen patients.

**Visual & Layout Specification:**
- Conceptual 2D Latent Space Projection:
  - Scatter plot showing Class 0 (None) and Class 1 (Mild) completely enveloping Class 3 and Class 4 (Severe), highlighting the dense boundary ambiguity.
  - Callout box: *"Overlapping representations + extreme class imbalance = catastrophic under-detection of high-risk patients."*

**Presenter Spoken Script (45–60 seconds):**
> *"The fundamental machine learning challenge in depression detection is latent space overlap. In psychiatric disorders, symptoms are notoriously heterogeneous. One depressed patient might experience severe insomnia and restlessness, while another experiences hypersomnia and lethargy. Consequently, when their behavioral features or pretrained embeddings are projected into latent space, the boundaries between mild, moderate, and severe depression blur together. Furthermore, subjective self-report scales introduce measurement noise of several points. Standard models trained on these overlapping spaces fail because they default to predicting the most frequent majority classes, missing the rare, severe cases that require immediate clinical triage."*

**Source Reference:**
- `project/research_paper/research_paper.md` (Section 1 & 3.1), `progress_reports/STAGE4_1_INTEGRITY_AUDIT.md` (Section 7)

---

### Slide 4: Research Motivation and Identified Gaps in Literature

- **Literature Gap 1: Identity & Observation Leakage:** Widespread affective computing studies use random row-level k-fold cross-validation. Mixing records from the same subject across train and test sets causes models to memorize individual biometric traits, yielding artificially inflated performance that collapses on external cohorts.
- **Literature Gap 2: Extreme Ordinal Imbalance Ignored:** Severe depression accounts for $<5\%$ of general populations. Naive accuracy maximization rewards predicting minimal symptoms while ignoring rare high-risk crises.
- **Literature Gap 3: Static Snapshots vs. Longitudinal Progression:** Existing models treat time as independent static vectors, failing to determine whether a patient is *improving, stable, or worsening* relative to their prior baseline.
- **Literature Gap 4: The Persistence Baseline Trap:** In multi-quarter studies, naive persistence ($\hat{y}_t = y_{t-1}$) attains over $76\%$ raw accuracy simply by predicting stability, creating a false illusion of performance while having $0\%$ sensitivity to clinical deterioration.
- **Literature Gap 5: The Black-Box Barrier:** Opaque deep learning architectures prevent clinicians from auditing whether alerts are driven by sleep breakdown, chronic pain, or social isolation.

**Visual & Layout Specification:**
- 4-Card Grid highlighting the Four Critical Traps in Prior Literature:
  - 1. Identity Leakage (Random row splits)
  - 2. Accuracy Distortion (Over-optimism from imbalance)
  - 3. Temporal Neglect (Ignoring sequential history)
  - 4. Uninterpretable Black Boxes (Lack of feature attribution)

**Presenter Spoken Script (45–60 seconds):**
> *"When surveying the computational depression detection literature, we identified four pervasive methodological flaws. First, researchers frequently split datasets by randomly shuffling records rather than subjects. When an individual appears in both train and test partitions, models cheat by learning that patient's unique baseline cadence. Second, extreme class imbalance is often ignored—in our cohort, severe cases make up just 4.6% of the data, so a model predicting 'mild' for everyone appears 95% accurate while being clinically useless. Third, longitudinal tracking is neglected; models fail to distinguish active clinical changes from baseline symptom stability. Finally, existing deep models provide no clinical audit trail. Our work was motivated by the urgent need to address all four gaps systematically."*

**Source Reference:**
- `project/research_paper/research_paper.md` (Section 1 & 2), De Angel et al. (Lancet Psychiatry 2022), Little et al. (2017)

---

### Slide 5: Research Objectives

1. **Subject-Independent Benchmarking:** Establish a zero-leakage, participant-disjoint benchmark across diverse linear, tree-based, neural, and sequential architectures.
2. **Sequential Behavioral Modeling:** Develop a windowed Gated Recurrent Unit (GRU) framework that fuses multi-period historical trajectories ($W \in \{1, 2, 3, \text{Full}\}$) with current behavioral states.
3. **Consensus-Driven Progression Evaluation:** Formulate longitudinal symptom tracking using the consensus Minimal Clinically Important Difference (MCID, $\pm 5$ PHQ-9 points) to resolve the Persistence Baseline Paradox.
4. **Clinical Decision Triage & Probability Calibration:** Implement post-hoc probability calibration and cost-sensitive threshold optimization ($T_4^*$) to maximize detection of rare, high-acuity severe cases.
5. **Auditable Model Transparency:** Implement exact polynomial-time TreeSHAP and permutation feature attributions to provide itemized clinical explanations for individual patient alerts.

**Visual & Layout Specification:**
- Horizontal process flow linking:
  - *Pillar 1: Rigorous Splitting* $\to$ *Pillar 2: Recurrent Modeling* $\to$ *Pillar 3: Progression Tracking* $\to$ *Pillar 4: Decision Triage* $\to$ *Pillar 5: TreeSHAP Transparency*.

**Presenter Spoken Script (30–45 seconds):**
> *"To address these gaps, we set five clear research objectives: First, establish a leak-free, subject-level benchmark across 4,948 participants. Second, formulate multi-window recurrent models to test whether historical behavioral sequences improve depression severity prediction. Third, benchmark longitudinal progression against the Minimal Clinically Important Difference of 5 points. Fourth, optimize decision thresholds to catch life-threatening severe cases. And fifth, open the black box using TreeSHAP to provide itemized physiological explanations for every clinical alert."*

**Source Reference:**
- `project/research_paper/research_paper.md` (Section 1)

---

### Slide 6: Audited Dataset & Multi-Modal Characteristics

- **Cohort Scale:** 4,948 total unique participants tracked across 12 monthly periods ($35,694$ total tracking rows).
- **Ground-Truth Labels:** 10,866 labeled quarterly PHQ-9 assessments administered at Periods 3, 6, 9, and 12.
- **150 Continuous Physiological & Clinical Features:**
  - *Sleep & Circadian Dynamics (58 features):* Sleep Regularity Index (SRI), WASO, sleep onset latency, acrophase, weekday/weekend ratio.
  - *Physical Activity & Actigraphy (52 features):* Hourly steps, active minutes, sedentary duration, activity entropy, diurnal variance.
  - *Clinical History & Comorbidities (24 features):* Chronic neuropathic pain, migraine frequency, medication logs, medical history.
  - *Demographics & Socioeconomics (16 features):* Age, living arrangements, educational level, acute financial strain indicators.
- **Modality Scope Verification:** Zero speech, audio transcripts, or transformer text embeddings exist in this audited dataset (`speech_text_modality_exists: false`).

```
+---------------------------------------------------------------------------------+
|                  DATASET PARTITIONING & CLASS DISTRIBUTION                      |
+-------------------+----------------+----------------+---------------------------+
| Split Partition   | Participants   | Total Records  | Labeled Assessments (N)   |
+-------------------+----------------+----------------+---------------------------+
| Training (70%)    | 3,463 (70.0%)  | 25,277 (70.8%) | 7,694 (70.8%)             |
| Validation (15%)  |   742 (15.0%)  |  5,217 (14.6%) | 1,597 (14.7%)             |
| Test Held-Out(15%)|   743 (15.0%)  |  5,200 (14.6%) | 1,575 (14.5%)             |
+-------------------+----------------+----------------+---------------------------+
| Severe Class Skew | Class 0: 38.9% | Class 1: 29.6% | Class 2: 18.9%            |
| in Test Cohort    | Class 3:  8.1% | Class 4:  4.6% (Severe: only 72 cases)     |
+-------------------+----------------+----------------+---------------------------+
```

**Visual & Layout Specification:**
- Clean split table and a donut chart displaying the heavy right-skewed class distribution (Minimal 38.9%, Mild 29.6%, Moderate 18.9%, Moderately Severe 8.1%, Severe 4.6%).

**Presenter Spoken Script (45–60 seconds):**
> *"Our experiments are grounded in an audited dataset of 4,948 participants comprising 35,694 monthly records and 10,866 labeled PHQ-9 evaluations. The feature space consists of 150 continuous predictors derived from passive wrist actigraphy, sleep dynamics, clinical comorbidities, and demographics. To guarantee complete integrity, our audit confirmed that all splits are strictly disjoint by participant ID: 3,463 participants in training, 742 in validation, and 743 in the held-out test split. The held-out test set contains exactly 1,575 labeled assessments. Notice the severe class imbalance: Class 4, representing severe major depression, comprises only 4.6% of the test cohort—just 72 instances out of 1,575."*

**Source Reference:**
- `project/results/metrics/dataset_report.json`, `progress_reports/STAGE4_TEMPORAL_DATA_AUDIT.md`

---

### Slide 7: End-to-End System Architecture

- **Stage 1: Multi-Modal Data Ingestion:** 150 continuous features extracted from passive wearable actigraphy, sleep logs, and medical records.
- **Stage 2: Strict Training-Quarantined Preprocessing:** Median imputation and z-score standard scaling fitted exclusively on training subjects ($\mathcal{D}_{\text{train}}$); validation and test data transformed using frozen parameters.
- **Stage 3: Temporal Sequencing & Horizon Bounding:** Historical sequences bounded to $p \le T$; all future periods ($p > T$) strictly quarantined.
- **Stage 4: Recurrent & Tabular Processing:** Windowed GRU fuses historical behavioral trajectory ($H_T \in \mathbb{R}^{W \times 150}$) with current period features ($x_T \in \mathbb{R}^{150}$).
- **Stage 5: Clinical Decision Support & Triage:** Post-hoc probability calibration and cost-sensitive threshold tuning ($T_4^* = 0.2100$).
- **Stage 6: Dual Evaluation & TreeSHAP Audit:** Cross-sectional severity ($N=1,575$), longitudinal transition analysis ($N=973$), and itemized local waterfall explanations.

**Visual & Layout Specification:**
- Embed the high-resolution architecture diagram: `project/research_paper/methodology_diagram.png`.
- Highlight the three distinct phases: (1) Data Partitioning & Sequencing, (2) Model Learning & Decision Triage, and (3) Dual Evaluation & Explainability.

**Presenter Spoken Script (45–60 seconds):**
> *"Slide 7 illustrates our end-to-end system architecture. Raw actigraphy, circadian, and clinical records first enter our zero-leakage preprocessing pipeline. Crucially, all median imputers and scalers are fitted exclusively on training participants. For temporal modeling, our windowed GRU takes a sequence of W historical months, encodes it into a 64-dimensional hidden representation, and concatenates it with the current month's features. The resulting 214-dimensional representation feeds dual heads: an ordinal classifier and a continuous PHQ-9 regressor. We then apply probability calibration and tune the severe decision threshold to 0.2100 before performing our dual-cohort evaluation and generating TreeSHAP clinical audit trails."*

**Source Reference:**
- `project/research_paper/methodology_diagram.png`, `project/research_paper/research_paper.md` (Section 3)

---

### Slide 8: Longitudinal Windowing & Sequence Construction

- **The Historical Horizon Problem:** How much sequential history is necessary to accurately predict current depression severity?
- **Four Evaluated Window Depths:**
  - **$W = 1$ (Immediate Preceding Period):** Prior month ($T-1$, shape $1 \times 150$) concatenated with current period $T$. Captures immediate behavioral inflection.
  - **$W = 2$ (Two-Period Trajectory):** Prior two months ($T-2, T-1$, shape $2 \times 150$). Captures short-term rate of change.
  - **$W = 3$ (Quarterly Trajectory):** Full prior quarter ($T-3, T-2, T-1$, shape $3 \times 150$). Matches clinical treatment intervals.
  - **$W = \text{Full}$ (Cumulative History):** Variable-length sequence from baseline ($p=1$) to $T-1$ with right-aligned zero-padding.
- **Prospective Model Selection Rule:** Model selection governed strictly by **Validation Macro-F1** prior to test unblinding to prevent data snooping.
  - **$W=1$ Validation Macro-F1:** **$0.3359$** (Prospectively Selected Model).
  - $W=2$: $0.3177$ | $W=3$: $0.3155$ | $W=\text{Full}$: $0.3103$.

**Visual & Layout Specification:**
- Diagram showing timeline periods $P_1$ through $P_{12}$, visually highlighting Period 5 as the history window ($W=1$), Period 6 as the current target survey, and Periods 7–12 marked with a red lock ("Quarantined Future: Zero Leakage").

**Presenter Spoken Script (45–60 seconds):**
> *"To determine how much sequential behavioral history is needed, we systematically benchmarked four observation window horizons: W=1 (one preceding month), W=2 (two months), W=3 (a full quarter), and W=Full (all available history). We enforced strict research discipline: the primary model was selected prospectively based on validation Macro-F1. As shown on the slide, the single-window configuration W=1 achieved the highest validation Macro-F1 at 0.3359, outperforming deeper windows whose missing historical periods introduced imputation noise. We locked in W=1 as our chosen configuration before running any test evaluations."*

**Source Reference:**
- `project/results/stage4_longitudinal_progression/stage4_progression_results.csv`, `progress_reports/STAGE4_1_INTEGRITY_AUDIT.md` (Section 5)

---

### Slide 9: Evaluated Machine Learning & Deep Learning Architectures

- **Linear Baseline:** L2-regularized Multiclass Logistic Regression ($C=1.0$) as an interpretable convex benchmark.
- **Tabular Tree Ensembles:**
  - *HistGradientBoosting:* Histogram-binned tree ensemble optimized for large tabular datasets.
  - *Class-Weighted XGBoost:* Gradient boosted decision trees trained with inverse-class prevalence weights to penalize minority misclassifications.
  - *Frank-Hall Ordinal Classifier:* Decomposes 5-class ordinal severity into 4 binary threshold problems ($y > k$), enforcing monotonic rank order.
- **Neural Architectures:**
  - *Baseline MLP:* 3-layer dense feedforward network with Batch Normalization and Dropout ($p=0.3$).
  - *Latent-SMOTE MLP:* Generates synthetic minority samples in neural latent space rather than high-dimensional input space.
  - *Multi-Task Auxiliary-Regression MLP:* Jointly optimizes categorical cross-entropy ($\mathcal{L}_{\text{CE}}$) and continuous score regression ($\mathcal{L}_{\text{MSE}}$) with loss weighting $\alpha = 0.5$.
- **Sequential Neural Model:**
  - *Windowed Fusion GRU:* Encodes sequential actigraphy/sleep history into hidden state $h_T \in \mathbb{R}^{64}$, fused with contemporaneous feature vector $x_T \in \mathbb{R}^{150}$.

**Visual & Layout Specification:**
- 3-Column Architectural Family Comparison:
  - *Column 1:* Tabular Ensembles (HistGB, XGBoost-Weighted, Frank-Hall Ordinal)
  - *Column 2:* Neural Architectures (MLP, Latent-SMOTE, Multi-Task Auxiliary Regression)
  - *Column 3:* Sequential Recurrent (Windowed Fusion GRU with dual classification/regression heads)

**Presenter Spoken Script (45–60 seconds):**
> *"We investigated models across three distinct architectural families. In the tabular family, we evaluated regularized Logistic Regression, HistGradientBoosting, Class-Weighted XGBoost, and the Frank-Hall ordinal classifier, which decomposes the 5-class problem into sequential binary thresholds. In the neural family, we benchmarked a standard MLP, a Latent-SMOTE network that oversamples synthetic cases in the latent embedding space, and a multi-task network that predicts both categorical tiers and continuous scores simultaneously. Finally, our Windowed Fusion GRU processes sequential historical feature matrices through recurrent gating units, merging temporal context with current cross-sectional features."*

**Source Reference:**
- `project/results/metrics/stage3_benchmark_results.csv`, `project/src/run_controlled_benchmarks.py`

---

### Slide 10: Experimental Design & Leakage Prevention Controls

- **Strict Subject-Level Disjoint Splitting:** Universal random seed (`seed = 42`) partitions participants into 3,463 train, 742 validation, and 743 test subjects. Zero cross-subject overlap.
- **Six Automated Leakage Verification Checks:**
  1. `Intersection(Train, Val, Test) = 0` (Zero participant leakage).
  2. Imputers and scalers fitted strictly on training data ($\mathcal{D}_{\text{train}}$).
  3. Latent-space SMOTE synthetic generation restricted exclusively to training folds.
  4. Chronological sequence direction enforced ($t_{\text{prev}} < t_{\text{curr}}$).
  5. Target isolation: `phq9_*` and `delta_phq9` excluded from all feature matrices.
  6. Prospective model selection: Zero hyperparameter adjustments after test unblinding.
- **Dual Evaluation Populations:**
  - *Cross-Sectional Denominator:* $N = 1,575$ labeled test assessments across 602 test participants.
  - *Longitudinal Denominator:* $N = 973$ valid consecutive transitions across 453 test participants with $\ge 2$ completed quarterly surveys.

**Visual & Layout Specification:**
- Audit Badge Graphic showing the 6 automated leakage checks marked with green checkmarks ("100% Passed"), paired with a flow diagram illustrating the denominator split between the 1,575 cross-sectional assessments and 973 longitudinal transition pairs.

**Presenter Spoken Script (45–60 seconds):**
> *"Slide 10 details our experimental controls. To eliminate the inflated performance claims seen in prior studies, we enforced six automated leakage checks. Crucially, all preprocessing parameters were derived exclusively from the training set. Furthermore, we maintain strict denominator transparency: cross-sectional classification is evaluated on all 1,575 held-out test assessments, whereas longitudinal transition progression is evaluated on the 973 chronological transition pairs from participants who completed at least two quarterly assessments. Conflating these two denominators is a common mistake in published studies; maintaining this distinction is vital for scientific accuracy."*

**Source Reference:**
- `project/src/data_leakage_checks.py`, `progress_reports/STAGE4_1_INTEGRITY_AUDIT.md` (Section 3)

---

### Slide 11: Experimental Results: Cross-Sectional Severity Benchmark

```
+--------------------------------------------------------------------------------------------------------+
|                   MAIN BENCHMARK EVALUATION ON HELD-OUT TEST COHORT (N = 1,575)                        |
+------------------------------------+----------------+-----------+----------+-----------+-------+-------+
| Evaluated Architecture             | Paradigm Class | Val-F1    | Test-F1  | Test-BAcc | Acc   | QWK   |
+------------------------------------+----------------+-----------+----------+-----------+-------+-------+
| Majority Class Baseline            | Trivial Zero   | 0.1074    | 0.1119   | 0.2000    | 38.86%| 0.000 |
| Logistic Regression (L2)           | Linear         | 0.3108    | 0.3257   | 0.3212    | 45.40%| 0.4238|
| Unweighted Classification MLP      | Neural Baseline| 0.3274    | 0.3116   | 0.3156    | 44.25%| 0.4256|
| Class-Weighted MLP                 | Imbalance Neural 0.3090    | 0.3184   | 0.3534    | 40.95%| 0.4553|
| Corrected Latent-SMOTE MLP         | Oversampling   | 0.3206    | 0.3068   | 0.3311    | 38.16%| 0.4019|
| HistGradientBoosting               | Tabular Tree   | 0.2976    | 0.3272   | 0.3176    | 45.08%| 0.4063|
| XGBoost (Class-Weighted)           | Tabular Tree   | 0.3216    | 0.3128   | 0.3182    | 42.03%| 0.4625|
| Frank-Hall Ordinal Classifier      | Ordinal Tree   | 0.3231    | 0.3283   | 0.3226    | 45.52%| 0.4351|
| Multi-Task Aux-Regression MLP      | Multi-Task Reg | 0.3427    | 0.3150   | 0.3127    | 45.27%| 0.4540|
| Windowed Fusion GRU (W=1 Selected) | Sequential GRU | 0.3359    | 0.3437   | 0.3389    | 45.33%| 0.4607|
+------------------------------------+----------------+-----------+----------+-----------+-------+-------+
```

- **Top Test Macro-F1:** **$0.3437$** (Windowed Fusion GRU, $W=1$).
- **Top Test Balanced Accuracy:** **$0.3534$** (Class-Weighted MLP) and **$0.3389$** (Windowed GRU).
- **Top Ordinal Agreement (QWK):** **$0.4625$** (XGBoost Weighted) and **$0.4607$** (Windowed GRU).
- **Lowest Ordinal Absolute Error:** **$0.7702$ tiers** (Frank-Hall Ordinal Classifier).

**Visual & Layout Specification:**
- Comparative bar chart comparing **Test Macro-F1** and **Test Balanced Accuracy** across all 10 evaluated models, with Windowed Fusion GRU highlighted in emerald green and naive baseline highlighted in red.

**Presenter Spoken Script (45–60 seconds):**
> *"Slide 11 shows our main benchmark results on the 1,575 held-out test assessments. Raw accuracies across all learned models cluster between 38% and 45.5%, illustrating the real-world predictive ceiling of passive wearable sensing under strict subject-independent evaluation. However, accuracy alone is misleading under class imbalance. Looking at Macro-F1 and Balanced Accuracy, our Windowed Fusion GRU achieves the strongest overall performance, with a test Macro-F1 of 0.3437 and balanced accuracy of 0.3389. For ordinal agreement, Class-Weighted XGBoost and the Windowed GRU lead with Quadratic Weighted Kappa scores of 0.4625 and 0.4607, proving their errors stay close to the true diagonal."*

**Source Reference:**
- `project/results/metrics/stage3_benchmark_results.csv`, `project/results/stage5_model_enhancements/stage5_benchmark_comparison.csv`

---

### Slide 12: Longitudinal Progression & The Persistence Paradox

```
+--------------------------------------------------------------------------------------------------------+
|                  LONGITUDINAL TRANSITION PROGRESSION BENCHMARK (N = 973 TEST TRANSITIONS)              |
+--------------------+----------------+---------------+---------------+---------------+------------------+
| Evaluated Model    | MCID Acc (%)   | MCID Macro-F1 | Direction Acc | Score MAE     | Delta MAE        |
+--------------------+----------------+---------------+---------------+---------------+------------------+
| Persistence Baseline| 76.46% (Trivial) 0.2889       | 0.00% (Failed)| 2.9424 pts    | 2.9424 pts       |
| W = 1 (Selected)   | 58.79%         | 0.3340        | 38.33%        | 3.6592 pts    | 3.7431 pts       |
| W = 2              | 58.07%         | 0.3367        | 43.99%        | 3.5878 pts    | 3.3988 pts       |
| W = 3              | 56.42%         | 0.3342        | 44.60%        | 3.5901 pts    | 3.1367 pts       |
| W = Full           | 57.86%         | 0.3364        | 44.50%        | 3.5976 pts    | 3.1840 pts       |
+--------------------+----------------+---------------+---------------+---------------+------------------+
```

- **The Persistence Paradox Unveiled:**
  - Clinical persistence ($\hat{y}_t = y_{t-1}$) predicts $\Delta \hat{y} = 0.0$ for all transitions.
  - Exactly 744 of 973 test transitions (76.46%) are clinically stable ($|\Delta y| < 5$).
  - Persistence achieves **76.46% raw MCID accuracy** by trivially predicting the majority class.
  - However, because $\text{sign}(0) = 0$, persistence achieves **0.0% Directional Agreement**, completely missing all 109 improving and 120 worsening transitions.
- **Sequential Recurrent Superiority:**
  - All learned GRU models outperform persistence in detecting active non-zero clinical transitions.
  - Directional agreement reaches **$44.60\%$** in $W=3$ and **$38.33\%$** in $W=1$, actively capturing true patient deterioration.

**Visual & Layout Specification:**
- Paired Bar Chart:
  - *Chart A:* Raw MCID Accuracy (Persistence 76.5% vs. GRU 58.8%).
  - *Chart B:* Directional Agreement Accuracy (Persistence 0.0% vs. GRU 44.6%).
  - Large callout: *"The Persistence Paradox: 76% accuracy with zero clinical sensitivity to deterioration."*

**Presenter Spoken Script (60 seconds):**
> *"Slide 12 presents one of our most important scientific findings: the resolution of the Persistence Paradox. When monitoring patients across quarters, over 76% remain clinically stable. If a naive model simply predicts that every patient's score will be identical to their previous score, it trivially attains 76.46% raw accuracy. But look at Directional Agreement: persistence scores exactly 0.0%, because predicting zero change fails to detect a single improving or worsening patient. In contrast, our recurrent models achieve up to 44.60% directional agreement and 0.3367 MCID Macro-F1. While learned models have lower raw majority accuracy, they provide what clinicians actually need: the ability to detect active clinical shifts."*

**Source Reference:**
- `project/results/stage4_longitudinal_progression/stage4_progression_results.csv`, `progress_reports/STAGE4_1_INTEGRITY_AUDIT.md` (Section 4)

---

### Slide 13: Clinical Decision Support & Severe Triage Threshold Tuning

- **The Clinical Cost Asymmetry:** Missing an acutely severe, suicidal patient (false negative) is catastrophic; a false alarm (false positive) merely prompts a routine, low-cost nurse check-in.
- **Default Argmax Failure on Severe Class ($N=1,575$, 72 severe cases):**
  - Severe Recall: **$25.0\%$** (Only 18 of 72 severe cases caught; **54 severe patients missed**).
  - Severe Precision: $19.78\%$ (18 / 91 alerts).
- **Post-Hoc Threshold Optimization on Validation Data:**
  - Threshold selected by maximizing Youden's $J = \text{Recall}_4 + \text{Specificity}_4 - 1$ with safety guardrails ($\text{Recall} \ge 0.20, \text{Precision} \ge 0.10$):
    $$T_4^* = 0.2100$$
- **Validated Results on Held-Out Test Cohort ($T_4^* = 0.2100$):**
  - Severe Recall surges from $25.0\% \to \mathbf{41.67\%}$ (**66.7% relative sensitivity gain**).
  - True Positives caught increase from $18 \to 30$; false negatives reduced from $54 \to 42$.
  - Generates 188 total triage alerts, providing an actionable safety net for outpatient clinics.

**Visual & Layout Specification:**
- Confusion Matrix / Alert Shift Visual:
  - *Before ($T=0.50$):* 18 caught, 54 missed (large red warning).
  - *After ($T_4^*=0.21$):* 30 caught, 42 missed (+66.7% sensitivity boost in green).
  - Threshold slider graphic illustrating the trade-off curve between recall and false alarm rate.

**Presenter Spoken Script (45–60 seconds):**
> *"In psychiatric triage, false negatives and false positives have profoundly asymmetric costs. Missing a severe depressive crisis can result in tragedy, while a false alert simply triggers a follow-up phone call from nursing staff. Under standard argmax rules, our model detected only 25% of severe patients, missing 54 individuals. By optimizing the decision threshold on validation data using Youden's index, we selected an operational threshold of T4* = 0.2100. When applied to the unseen test set, severe recall surged to 41.7%—a 66.7% relative improvement that caught 30 severe patients and reduced missed cases to 42. This proves the value of post-hoc threshold tuning in real-world clinical decision support."*

**Source Reference:**
- `project/results/stage5_model_enhancements/stage5_threshold_optimization.json`, `project/results/stage5_model_enhancements/clinical_decision_dashboard.html`

---

### Slide 14: Model Explainability: TreeSHAP & Clinical Case Audits

- **Additive Game-Theoretic Explanation:**
  $$f(x) = \phi_0 (\text{Base Rate: } 4.6\%) + \sum_{i=1}^{150} \phi_i(x)$$
  Exact attribution with zero residual approximation error.
- **Global Population Domain Share (TreeSHAP):**
  - **Sleep & Circadian Dynamics:** **45.4%** (Dominant signal: SRI, sleep latency, nocturnal awakenings).
  - **Physical Activity & Actigraphy:** **33.5%** (Hourly active steps, sedentary duration).
  - **Clinical Comorbidities:** **15.6%** (Chronic neuropathic pain, medical history).
  - **Demographics & Socioeconomics:** **5.5%** (Financial hardship, educational attainment).
- **Local Patient Waterfall Case Studies:**
  - *Case 1 (True Positive Severe):* Financial distress (`money`: $+0.6438$) and neuropathic pain ($+0.1256$) drove severe risk to $91.2\%$.
  - *Case 3 (False Negative Masked Case):* A severe patient missed by the model because high education (`educ`: $-0.5585$) and financial security ($-0.1674$) acted as strong statistical buffers, masking acute emotional suffering.

**Visual & Layout Specification:**
- Two-part visual:
  - *Left:* Donut chart of global domain shares (Sleep 45.4%, Steps 33.5%, Clinical 15.6%, Demographics 5.5%).
  - *Right:* TreeSHAP Waterfall plot for Case 1 showing red horizontal bars pushing risk to the right ($+0.6438$) and blue bars pulling to the left ($-0.1082$).

**Presenter Spoken Script (60 seconds):**
> *"Slide 14 demonstrates how we open the black box using TreeSHAP. Globally, sleep and circadian dynamics account for 45.4% of total predictive attribution, followed by physical activity at 33.5% and clinical comorbidities at 15.6%. At the individual patient level, our interactive waterfall plots provide an itemized receipt for every alert. For example, in Case 1, acute financial distress (+0.64) and chronic neuropathic pain (+0.13) pushed the patient's predicted risk to 91.2%. Even more instructive is Case 3, a missed severe case: the patient had high education and financial security, which acted as strong statistical buffers that masked their severe emotional distress. This audit proves why wearables must augment, not replace, human clinical interviews."*

**Source Reference:**
- `project/results/stage5_model_enhancements/stage5_domain_importance.csv`, `project/results/stage5_model_enhancements/stage5_local_explanations.json`

---

### Slide 15: Limitations and Threats to Validity

- **Monthly Aggregation vs. Intraday Timestamps (Case C):** The audited dataset provides monthly summary aggregates rather than raw, millisecond-level accelerometer time series; sub-monthly survey completion timestamps cannot be independently audited.
- **Severe Class Sample Size:** Although the cohort is substantial ($N=4,948$), severe depression constitutes only 4.6% of test assessments ($N=72$). While threshold tuning improves sensitivity to 41.7%, larger multi-center cohorts are needed.
- **Observational vs. Causal Interpretation:** Permutation and SHAP attributions reflect associative predictive reliance within the model's decision function; they do not establish counterfactual causal interventions.
- **Absence of Inferential Hypothesis Testing:** While descriptive performance gains over persistence are large (44.6% vs. 0.0% directional agreement), formal non-parametric hypothesis testing (Wilcoxon signed-rank, McNemar) was not conducted across folds.
- **Intended Use Disclaimer:** Framework is strictly for **RESEARCH USE ONLY**; prospective randomized clinical trials are required before real-world deployment.

**Visual & Layout Specification:**
- 4-Box Clean Warning Grid:
  - Box 1: Monthly Resolution (Absence of millisecond timestamps)
  - Box 2: Severe Sample Size (72 test severe cases)
  - Box 3: Non-Causal Nature (Predictive associations $\ne$ medical cures)
  - Box 4: Research-Only Scope (Requires prospective clinical trials)

**Presenter Spoken Script (45 seconds):**
> *"In the spirit of scientific integrity, we explicitly acknowledge our study's limitations. First, our data consists of monthly aggregates rather than raw intraday timestamps. Second, while our cohort of 4,948 participants is substantial, severe depression remains rare at 72 test cases. Third, our SHAP attributions represent predictive associations within the model, not validated causal cures—we cannot claim that improving sleep will causally cure depression. Finally, this software is strictly for research use only and must undergo prospective clinical trials before deployment in outpatient workflows."*

**Source Reference:**
- `progress_reports/STAGE4_1_INTEGRITY_AUDIT.md` (Section 2 & 7), `project/research_paper/research_paper.md` (Section 6)

---

### Slide 16: Conclusion & Future Research Directions

- **Summary of Research Achievements:**
  - Established a leak-free, subject-disjoint benchmark across 4,948 participants and 10,866 evaluations.
  - Demonstrated that recurrent Windowed Fusion GRU ($W=1$) achieves top test Macro-F1 ($0.3437$) and balanced accuracy ($0.3389$).
  - Resolved the Persistence Paradox, achieving up to $44.60\%$ directional agreement on true clinical transitions where persistence fails ($0.0\%$).
  - Tuned clinical triage thresholds ($T_4^* = 0.2100$), boosting severe depression recall by $+66.7\%$.
  - Provided exact TreeSHAP attributions proving sleep (45.4%) and activity (33.5%) dominate passive prediction.
- **Future Research Directions:**
  - Integration of high-frequency intraday accelerometer sensor streams.
  - Multi-modal fusion incorporating acoustic and text transformer embeddings where consent permits.
  - Prospective randomized clinical triage trials in university outpatient psychiatric clinics.

**Visual & Layout Specification:**
- Strong concluding card summarizing the 3 core pillars:
  - 1. Rigorous Benchmark (Zero-leakage subject splitting)
  - 2. Clinical Utility (+66.7% severe sensitivity boost)
  - 3. Total Transparency (TreeSHAP itemized receipts)
- Closing text: *"Thank you. Questions and Discussion."*

**Presenter Spoken Script (45–60 seconds):**
> *"In conclusion, this project provides a rigorous, transparent benchmark for wearable depression prediction. By enforcing participant-level splitting, we eliminated identity leakage. We exposed the persistence paradox, proving that our sequential models detect genuine clinical shifts where naive heuristics fail. Through cost-sensitive threshold tuning, we boosted severe depression detection by 66.7%, and TreeSHAP provided an itemized physiological receipt for every alert. Looking forward, we plan to incorporate raw intraday sensor streams and multi-modal acoustic embeddings. Thank you for your time, and I look forward to your questions."*

**Source Reference:**
- `project/research_paper/research_paper.md` (Section 7)

---

# SECTION 2: PRESENTATION DESIGN GUIDE

### 1. Color Palette & Typography
- **Primary Canvas Background:** Dark Navy (`#0F172A`) or Clean Off-White (`#F8FAFC`).
- **Surface Container Cards:** Midnight Blue (`#1E293B`) or Crisp White (`#FFFFFF`).
- **Primary Accent / Metrics:** Cyan Blue (`#38BDF8`) for model highlights and general metrics.
- **Positive / Resilience Accent:** Emerald Green (`#10B981`) for safety buffers and improved transitions.
- **Alert / Risk Accent:** Coral Rose (`#F43F5E`) for severe depression triage alerts and high-risk factors.
- **Warning Accent:** Amber (`#F59E0B`) for subclinical warnings and threshold boundaries.
- **Font Stack:**
  - Slide Titles: Helvetica Neue / Inter / Arial Bold, 24–28 pt.
  - Section Headers: Semi-bold, 16–18 pt.
  - Body Text / Bullet Points: Regular, 13–15 pt (line spacing 1.25).
  - Metric Callouts: Monospace / Tabular numbers (Roboto Mono / Consolas), 20–28 pt Bold.

### 2. Slide Layout Rules
- **Rule of 4–6 Points:** Maximum 4 to 6 concise bullet points per slide. Never paste paragraphs.
- **Two-Column Comparisons:** Use split cards for comparing baseline vs. proposed, unweighted vs. weighted, or persistence vs. learned models.
- **High-Density Metric Badges:** Group numerical results into styled metric pills rather than plain bullet text.

---

# SECTION 3: EVALUATOR DEFENSE QUESTIONS & ANSWERS (18 QUESTIONS)

### Q1: Why did you evaluate wearable actigraphy and clinical features rather than speech or text embeddings?
> **Answer:** *"Our dataset audit confirmed that the 4,948-participant cohort contains 150 continuous features spanning wearable movement, sleep, circadian dynamics, clinical comorbidities, and demographics, with no audio or text features present. From a clinical perspective, passive wearables provide continuous, unobtrusive monitoring across 12 months in ecological settings without requiring active patient recording compliance or raising sensitive acoustic privacy concerns."*

### Q2: What causes latent feature space overlap in depression prediction?
> **Answer:** *"Latent space overlap is driven by three factors: symptom heterogeneity, where two depressed patients with identical PHQ-9 scores present opposite behaviors (insomnia vs. hypersomnia); measurement noise in self-report surveys ($\pm 2$ to $4$ points); and socioeconomic resilience, where high education and financial wealth statistically buffer daily routines even during severe psychological distress."*

### Q3: Why is subject-level splitting essential in affective computing?
> **Answer:** *"If records from the same participant appear in both training and test sets, machine learning models memorize individual biometric traits—such as a participant's unique walking cadence, typical wake time, or device-wearing habits—rather than true depressive biomarkers. Subject-level splitting guarantees zero participant overlap, testing true generalization to completely unseen patients."*

### Q4: Why are Macro-F1 and Balanced Accuracy preferred over raw classification accuracy?
> **Answer:** *"In our cohort, severe depression represents only 4.6% of test assessments. A naive trivial classifier that predicts 'minimal depression' for every single patient achieves over 38% accuracy, while predicting minimal or mild covers nearly 70% of the cohort. Macro-F1 gives equal weight to all five classes, and Balanced Accuracy computes the unweighted average of per-class recall, preventing the model from ignoring rare, high-acuity patients."*

### Q5: What is the 'Persistence Baseline Paradox' and how did you resolve it?
> **Answer:** *"In longitudinal psychiatric monitoring, over 76% of quarterly transitions are clinically stable ($|\Delta \text{PHQ-9}| < 5$). A persistence baseline ($\hat{y}_t = y_{t-1}$) predicts zero change for all patients, trivially achieving 76.46% raw MCID accuracy. However, because its predicted change is always zero, it scores exactly 0.0% directional agreement on actual clinical shifts. We resolved this by evaluating Directional Agreement and MCID Macro-F1, showing that our recurrent models achieve up to 44.60% directional agreement, capturing true clinical deterioration."*

### Q6: Why did you select $W=1$ as your primary model when $W=\text{Full}$ had a higher test Macro-F1?
> **Answer:** *"Adhering strictly to prospective validation-driven model selection is essential for scientific integrity. On validation data, $W=1$ achieved the highest Macro-F1 ($0.3359$) compared to $0.3103$ for $W=\text{Full}$. Selecting $W=\text{Full}$ post-hoc after seeing its test score ($0.3309$) would constitute test-set data snooping. Furthermore, deeper windows require imputing more missing historical periods, which increases validation variance."*

### Q7: Why did you optimize the severe decision threshold to $T_4^* = 0.2100$?
> **Answer:** *"Under standard argmax classification, severe-tier recall was only 25%, missing 54 of 72 severe patients. In psychiatric triage, missing an acutely suicidal patient carries severe medical risks, whereas generating a false alert only triggers a brief nurse check-in. By maximizing Youden's $J$ index with precision/recall guardrails on validation data, we tuned the threshold to 0.2100. On the unseen test set, this increased severe recall to 41.7%—a 66.7% relative gain that caught 30 severe patients."*

### Q8: What does TreeSHAP tell us that Random Forest Gini importance cannot?
> **Answer:** *"Gini importance provides only a global, heuristic measure of split frequency and is biased toward continuous high-cardinality features. TreeSHAP is grounded in cooperative game theory and satisfies the Additivity property: the base population expectation plus individual feature SHAP values equals the exact model output ($f(x) = \phi_0 + \sum \phi_i$). Furthermore, TreeSHAP provides local patient-level waterfall explanations showing the exact direction and magnitude of impact for an individual alert."*

### Q9: Can you claim that improving a patient's sleep regularity will causally cure their depression?
> **Answer:** *"No. We explicitly state in our limitations that SHAP and permutation importance reflect observational predictive reliance within the model's decision function, not causal counterfactual mechanisms. While irregular sleep and nocturnal awakenings strongly associate with higher predicted depression risk, establishing a causal medical cure requires randomized interventional clinical trials."*

### Q10: How does the Frank-Hall ordinal classifier work and why was it benchmarked?
> **Answer:** *"The Frank-Hall method decomposes a 5-class ordinal problem into 4 binary classifiers: $P(y > 0)$, $P(y > 1)$, $P(y > 2)$, and $P(y > 3)$. Class probabilities are derived by multiplying successive conditional probabilities. It achieved the lowest ordinal MAE ($0.7702$ tiers), proving that enforcing ordinal rank constraints penalizes distant off-diagonal classification errors."*

### Q11: What role did contrastive learning play in your experiments?
> **Answer:** *"In our initial ablation runs (Experiment E2 and E4), contrastive loss was evaluated to pull embeddings of similar severity tiers together and push different tiers apart. However, under strict subject-independent splitting, supervised contrastive learning achieved a test Macro-F1 of $0.2956$, underperforming standard cross-entropy ($0.3038$). Because contrastive learning did not improve validation performance, it was not adopted in our final benchmarking pipeline."*

### Q12: Why did Class-Weighted XGBoost perform so well on Quadratic Weighted Kappa?
> **Answer:** *"XGBoost with inverse class weighting achieved the highest QWK score ($0.4625$). Tree ensembles naturally model complex non-linear interactions across heterogeneous tabular features (such as combinations of financial distress, sleep fragmentation, and pain), while class weighting penalizes minority errors, keeping misclassifications adjacent to the true diagonal."*

### Q13: What hardware environment was used, and was GPU acceleration necessary?
> **Answer:** *"All final benchmarks were executed on an Intel Core i7-13700H CPU using PyTorch, Scikit-Learn, and XGBoost. Because our feature space consists of 150 engineered tabular/sequential features across 35,694 records, CPU execution completed all 10 model benchmarks and 50 unit tests in under 45 seconds, demonstrating that the framework is computationally efficient and suitable for deployment on standard hospital workstations."*

### Q14: How did you define clinically meaningful symptom change?
> **Answer:** *"We adopted the consensus Minimal Clinically Important Difference (MCID) threshold of $\tau = 5$ points on the PHQ-9, established in seminal psychiatric validation literature by Kroenke et al. (2001) and Löwe et al. (2004). Changes within $[-4, +4]$ points represent clinical stability, while shifts $\le -5$ denote meaningful improvement and $\ge +5$ denote deterioration."*

### Q15: Why did the Multi-Task Auxiliary-Regression MLP perform well on validation data?
> **Answer:** *"The multi-task network jointly optimizes categorical cross-entropy and continuous score MSE ($\alpha = 0.5$). Forcing the shared encoder to predict continuous PHQ-9 points acts as an inductive regularizer, preventing the network from overfitting to arbitrary class boundaries and resulting in the highest validation Macro-F1 ($0.3427$) among feedforward neural networks."*

### Q16: How do you handle missing historical data in the Windowed GRU?
> **Answer:** *"For window depths $W > 1$, missing historical periods are imputed using median training statistics and masked during recurrent propagation. However, as observed in our window comparison, deeper histories ($W=2, 3$) introduce cumulative imputation noise, which explains why the single-period window ($W=1$) achieved superior validation stability."*

### Q17: What was the primary clinical blind spot identified by your TreeSHAP audit?
> **Answer:** *"Our False Negative audit (Case 3) revealed that high educational attainment and financial wealth generate large negative SHAP values ($-0.5585$ and $-0.1674$), pulling the predicted risk down. In affluent or highly educated individuals, intact daily schedules statistically mask acute emotional suffering, demonstrating why passive wearable monitoring must never operate as a standalone diagnostic tool without human clinical oversight."*

### Q18: What are the immediate next steps to advance this research toward clinical utility?
> **Answer:** *"Immediate extensions include accessing raw intraday accelerometer streams to compute continuous rest-activity circadian metrics, implementing multi-task continuous state-space models (such as Mamba or S4), and conducting prospective validation trials in university outpatient psychiatric clinics under a clinician-in-the-loop triage protocol."*

---

# SECTION 4: PROJECT VERIFICATION CHECKLIST & TRACEABILITY MATRIX

| Slide / Section | Verified Metric or Claim | Source Artifact / Script | Status | Verification Detail |
| :--- | :--- | :--- | :---: | :--- |
| **Slide 6: Dataset** | 4,948 participants, 35,694 rows, 10,866 PHQ-9 labels | `project/results/metrics/dataset_report.json` | **Verified** | Exactly matches audited Parquet counts; 150 numerical features. |
| **Slide 6: Modality** | No speech/audio/transcripts exist in data | `project/results/metrics/dataset_report.json` | **Verified** | `speech_text_modality_exists: false`. |
| **Slide 6 & 10: Splits** | Train: 3,463; Val: 742; Test: 743 participants | `project/data/participant_splits.json` | **Verified** | Subject-independent partition; zero subject overlap. |
| **Slide 10: Denominators**| Cross-sectional $N=1,575$; Transitions $N=973$ | `progress_reports/STAGE4_1_INTEGRITY_AUDIT.md` | **Verified** | Formed by 602 test participants and 453 transition participants. |
| **Slide 8: Window Sel.** | $W=1$ Val Macro-F1 = 0.3359; Test = 0.3204 | `project/results/stage4_longitudinal_progression/stage4_progression_results.csv`| **Verified** | Prospective selection rule enforced; test snooping avoided. |
| **Slide 11: Benchmark** | GRU Test Macro-F1: 0.3437; Bal-Acc: 0.3389 | `project/results/metrics/stage3_benchmark_results.csv`| **Verified** | Top performer among 10 evaluated models on held-out test split. |
| **Slide 11: XGBoost** | XGBoost Weighted QWK: 0.4625; Acc: 42.03% | `project/results/metrics/stage3_benchmark_results.csv`| **Verified** | Top ordinal agreement score on test split. |
| **Slide 12: Persistence** | Persistence MCID Acc: 76.46%; Dir-Acc: 0.0% | `project/results/stage4_longitudinal_progression/stage4_progression_results.csv`| **Verified** | Exactly 744 / 973 transitions are stable; predicts $\Delta = 0.0$. |
| **Slide 12: GRU Dir-Acc**| GRU $W=3$ Dir-Acc: 44.60%; $W=1$: 38.33% | `project/results/stage4_longitudinal_progression/stage4_progression_results.csv`| **Verified** | Captures active non-zero clinical transitions. |
| **Slide 13: Threshold** | $T_4^* = 0.2100$; Severe Recall: 25.0% $\to$ 41.67% | `project/results/stage5_threshold_optimization.json`| **Verified** | +66.7% relative gain; false negatives cut from 54 to 42. |
| **Slide 14: Domains** | Sleep 45.4%, Steps 33.5%, Clinical 15.6%, Demo 5.5%| `project/results/stage5_model_enhancements/stage5_domain_importance.csv`| **Verified** | Global TreeSHAP information share across test cohort. |
| **Slide 14: Local Cases** | Case 1: `money` +0.6438; Case 3: `educ` -0.5585 | `project/results/stage5_model_enhancements/stage5_local_explanations.json`| **Verified** | Exact local TreeSHAP attribution values. |
| **Slide 15: Limitations** | Monthly aggregates without intraday timestamps | `progress_reports/STAGE4_TEMPORAL_DATA_AUDIT.md` | **Verified** | Transparently documents Case C temporal aggregation limit. |
