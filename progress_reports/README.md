# 📈 Longitudinal Depression Prediction: Progress Report Tracker

This directory maintains the chronological history and version-controlled progress reports for all model iterations, experimental runs, and code changes across the project.

---

## 🎯 Progress Reports Index

| Report ID | Date | Key Code Changes / Focus | Best Test Acc | Best Macro-F1 | Best Bal Acc | Best Score MAE | Status / Notes |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **[Report 1](PROGRESS_REPORT_1.md)** | Oct 2026 | Initial baseline pipeline (E1–E8), zero-leakage subject split, latent SMOTE, SupCon, multimodal fusion | **45.27%** (E5) | **0.3170** (E5) | **0.3205** (E3) | **3.9314** (E1) | Benchmark Established |
| **[Report 2](PROGRESS_REPORT_2.md)** | Oct 2026 | Complete code audit, fixed current-observation duplication in history, variable-length GRU masking, 14-test verification suite, clean classification MLP, Class-Weighting, Ordinal Classification, and Multi-task auxiliary grid | **45.90%** (Exp 7) | **0.3427** (Exp 7 Val) / **0.3468** (Exp 7 Test) | **0.3727** (Exp 4a Val) / **0.3626** (Exp 4a Test) | **3.9314** (Exp 1) | **Empirically Verified & Leakage-Repaired** |
| **[Report 3](PROGRESS_REPORT_3.md)** | Oct 2026 | Controlled benchmark grid (11 models), 26-test verification suite, longitudinal matched cohort ($N=1,523$), feature ablations, multi-seed stability (seeds 42, 123, 2026), SHAP & t-SNE | **45.52%** (Frank–Hall) | **0.3427** (Aux-Reg Val) / **0.3437** (Current GRU Test) | **0.3625** (Class-Weighted Val) / **0.3534** (Class-Weighted Test) | **3.9413** (Aux-Reg) | **Audited & Stage 3.1 Verified** |
| **[Stage 4 Report](STAGE4_LONGITUDINAL_PROGRESSION_REPORT.md)** | Oct 2026 | Longitudinal severity progression, temporal audit (Case C), configurable windows ($W \in \{1, 2, 3, \text{Full}\}$), MCID 3-class progression, within-subject $\Delta y_t$ tracking, individual trajectories & temporal explainability | **42.73%** (W=Full) | **0.3359** (W=1 Val) / **0.3309** (W=Full Test) | **0.3511** (W=Full Test) | **3.8921** (W=Full) / **3.1435** ($\Delta$ MAE) | **Completed & Fully Verified** |
| **[Stage 5 Report](STAGE5_EXPERIMENT_REPORT.md)** | Oct 2026 | Controlled model enhancement, strong tabular benchmark (XGBoost), Policy A vs. Policy B (Prior-PHQ-9), severe threshold optimization ($T_4^*=0.21$), Platt/Isotonic calibration, TreeSHAP, continuous regression & interactive dashboard | **48.51%** (Policy B) / **45.33%** (Policy A) | **0.3976** (Policy B Val) / **0.3901** (Policy B Test) | **0.3991** (Policy B Test) / **0.3301** (Policy A Calib) | **3.9143** (Reg MAE) | **Completed & Fully Verified** |

---

## 📋 Iteration Protocol for Future Runs

Whenever any code change is made that impacts model performance (increases or decreases metrics):
1. **Never overwrite previous reports:** Existing reports remain immutable snapshots.
2. **Create the next sequential report:** `PROGRESS_REPORT_2.md`, `PROGRESS_REPORT_3.md`, etc.
3. **Document the Delta ($\Delta$):**
   - Exact code/architectural changes made (files modified, lines changed).
   - Direct comparison table comparing the new run against previous iterations.
   - Analysis of *why* accuracy/F1/MAE shifted.
   - Actionable takeaways and next steps.
