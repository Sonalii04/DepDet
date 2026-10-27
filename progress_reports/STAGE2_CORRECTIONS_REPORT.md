# 🛠️ Stage 2 Report: Approved Corrections & Methodological Hardening

- **Project:** Explainable Contrastive Longitudinal Depression Prediction Framework
- **Stage:** Stage 2 (Approved Corrections)
- **Status:** Completed & Empirically Verified
- **Test Suite Result:** 26 / 26 Unit Tests PASSED (`Ran 26 tests in 17.829s ... OK`)
- **Backups Preserved:** [project/src/backups/stage1_backup/](file:///d:/projects/DepDet/project/src/backups/stage1_backup)

---

## 1. Summary of Approved Corrections by Task

### Task 1 — Standardized Evaluation Metrics
- **Target File:** [evaluation.py](file:///d:/projects/DepDet/project/src/evaluation.py)
- **Modifications Implemented:**
  1. **Standard Metrics:** Accuracy, balanced accuracy, macro and weighted precision/recall/F1.
  2. **Per-Class Metrics:** Per-class precision, recall, and F1 explicitly dictionary-mapped for classes `0, 1, 2, 3, 4`.
  3. **Multiclass G-Mean:** Implemented `compute_multiclass_g_mean` with the documented convention: if any class recall is 0.0, G-Mean is strictly 0.0 (preventing starvation of minority severity classes).
  4. **Ordinal Metrics:** Quadratic Weighted Kappa (`quadratic_weighted_kappa`) and ordinal class-index MAE (`ordinal_class_mae`).
  5. **Probability Metrics:** Macro One-vs-Rest ROC-AUC (`roc_auc_ovr_macro`) and PR-AUC (`pr_auc_ovr_macro`) whenever valid probability matrices are supplied.
  6. **Confusion Matrices:** Standardized fixed class ordering `[0, 1, 2, 3, 4]`.
  7. **Edge Cases:** All absent classes, division-by-zero, and single-class distributions handled explicitly with `zero_division=0` and exception safety.

### Task 2 — Correct Latent-SMOTE Implementation
- **Target Files:** [oversampling.py](file:///d:/projects/DepDet/project/src/oversampling.py) & [run_controlled_benchmarks.py](file:///d:/projects/DepDet/project/src/run_controlled_benchmarks.py)
- **Modifications Implemented:**
  1. Replaced the uninitialized/untrained MLP encoder in `oversampling.py` with a genuinely trained unweighted baseline MLP fitted strictly on training data.
  2. Decoupled latent SMOTE in `run_controlled_benchmarks.py` (Exp 4) so that embeddings are generated strictly by the clean unweighted MLP (`model_e2`), eliminating the methodological coupling with the class-weighted MLP.
  3. Preserved strict data isolation: validation and test observations are never exposed to SMOTE interpolation or resampling.

### Task 3 — Correct Longitudinal Regression Training
- **Target File:** [longitudinal.py](file:///d:/projects/DepDet/project/src/longitudinal.py)
- **Modifications Implemented:**
  1. Integrated continuous PHQ-9 regression training using masked MSE loss: `loss = loss_c + regression_weight * loss_r`.
  2. Implemented valid target masking (`~torch.isnan(y_score)`) so missing continuous scores do not corrupt backpropagation.
  3. Verified gradient flow and parameter updates on `model.reg_score` during training.
  4. Maintained checkpoint selection and early stopping strictly on validation classification Macro-F1 (the primary objective).

### Task 4 — Correct GRU Padding Handling
- **Target File:** [multitask.py](file:///d:/projects/DepDet/project/src/multitask.py)
- **Modifications Implemented:**
  1. Aligned variable-length sequence handling in `MultiTaskFullModel` with the verified length-indexed approach: `gru_out[b, lengths[b]-1, :]`.
  2. Implemented `self.no_history_emb` for instances with `lengths[b] == 0`, ensuring trailing zero-padding cannot alter the encoded patient history representation.

### Task 5 — XGBoost Environment and Implementation
- **Target File:** [xgboost_model.py](file:///d:/projects/DepDet/project/src/xgboost_model.py)
- **Environment Status:** Installed `xgboost 3.4.1` successfully into project Python environment (`C:\Python313\python.exe`).
- **Modifications Implemented:**
  1. Implemented `build_xgboost_classifier` with `objective="multi:softprob"`, `num_class=5`, `eval_metric="mlogloss"`, and early stopping.
  2. Implemented `compute_multiclass_sample_weights` calculated strictly from training labels ($w_c = N / (C \cdot N_c)$).
  3. Implemented `train_xgboost_model` supporting subject-level train/val/test splits, model evaluation, and saving versioned metrics.

---

## 2. Unit Test Suite Execution & Verification

Executed via:
```bash
python -m unittest project/src/test_pipeline_and_leakage.py
```

### Complete Test Results (26 Tests)

| Category / Suite | Test Case | Status | Details |
| :--- | :--- | :---: | :--- |
| **Pipeline & Leakage Integrity** | `test_condition_1_no_participant_in_multiple_splits` | **PASSED** | Zero participant overlap across train/val/test |
| | `test_condition_2_no_target_column_enters_feature_matrix` | **PASSED** | Target leakage strictly checked |
| | `test_condition_3_no_future_observation_enters_history` | **PASSED** | Temporal order preserved |
| | `test_condition_4_current_observation_not_duplicated_in_history` | **PASSED** | Period 1 history length is 0 |
| | `test_condition_5_padded_time_steps_do_not_affect_gru_output` | **PASSED** | Trailing padding invariance verified |
| | `test_condition_6_phq9_categories_match_documented_score_ranges` | **PASSED** | Cutoffs [0-4, 5-9, 10-14, 15-19, 20-27] exact |
| | `test_condition_7_delta_targets_use_documented_definition` | **PASSED** | Delta target calculation exact |
| | `test_condition_8_preprocessing_fitted_on_training_data_only` | **PASSED** | Fitted statistics unaffected by val/test |
| | `test_condition_9_class_weights_computed_from_training_labels_only` | **PASSED** | Weighting verified on train labels |
| | `test_condition_10_validation_and_test_never_resampled` | **PASSED** | Val and test sample counts constant |
| | `test_condition_11_evaluation_metrics_handle_absent_predicted_classes` | **PASSED** | Zero division handled without crash |
| | `test_condition_12_repeated_runs_reproducible_with_seed` | **PASSED** | Reproducibility confirmed |
| | `test_condition_13_checkpoint_loading_restores_intended_model` | **PASSED** | State reload verified |
| | `test_condition_14_evaluation_code_does_not_train_on_test_data` | **PASSED** | Eval mode zero parameter drift |
| **Task 1: Evaluation Metrics** | `test_hand_checkable_perfect_classification` | **PASSED** | Acc=1.0, F1=1.0, QWK=1.0, MAE=0.0, G-Mean=1.0 |
| | `test_hand_checkable_known_imperfect_classification` | **PASSED** | Verified Ordinal MAE = 0.1, CM shape (5,5) |
| | `test_multiclass_g_mean_zero_class_recall` | **PASSED** | Returns 0.0 when any class recall is 0.0 |
| | `test_ovr_roc_auc_and_pr_auc_valid_probabilities` | **PASSED** | OVR ROC-AUC and PR-AUC Macro computed correctly |
| | `test_confusion_matrix_fixed_class_order` | **PASSED** | Confusion matrix maintains 5x5 fixed ordering |
| **Task 2: Latent SMOTE** | `test_latent_smote_resampling_balances_classes` | **PASSED** | Minority classes balanced to target count |
| | `test_validation_and_test_never_enter_smote` | **PASSED** | Validation & test latents strictly untouched |
| **Task 3: Longitudinal Regression**| `test_regression_loss_computation_and_parameter_update` | **PASSED** | Non-zero MSE loss, gradients reach `reg_score`, weights update |
| | `test_missing_regression_target_masking` | **PASSED** | NaN regression targets safely masked out |
| **Task 4: Multitask Padding** | `test_multitask_gru_padding_invariance` | **PASSED** | Hidden state bitwise invariant to trailing padding |
| | `test_multitask_no_history_handling` | **PASSED** | Zero history handled safely with `no_history_emb` |
| **Task 5: XGBoost Classifier** | `test_xgboost_output_shape_probabilities_and_class_order`| **PASSED** | Probs shape (N, 5), sum to 1.0, class order [0..4] |

```
Ran 26 tests in 17.829s
OK
```

---

## 3. Inventory of Modified Files

1. `project/src/evaluation.py`: Added multiclass G-mean, per-class metrics, QWK, ordinal MAE, OVR ROC-AUC, OVR PR-AUC, and fixed 5x5 confusion matrix order.
2. `project/src/oversampling.py`: Replaced uninitialized MLP with trained unweighted MLP for latent space SMOTE.
3. `project/src/run_controlled_benchmarks.py`: Passed unweighted clean encoder (`model_e2`) into latent SMOTE, decoupling from class-weighted MLP.
4. `project/src/longitudinal.py`: Added continuous regression loss term (`regression_weight * MSE`), missing target masking, and gradient supervision on `reg_score`.
5. `project/src/multitask.py`: Aligned variable-length GRU indexing and added `no_history_emb` for zero-history cases.
6. `project/src/xgboost_model.py` (New): Implemented XGBoost multiclass model, training-only sample weights, and early stopping.
7. `project/src/test_pipeline_and_leakage.py`: Expanded with 12 new targeted unit tests for Tasks 1–5, bringing total verified conditions from 14 to 26.
8. `project/src/backups/stage1_backup/*`: Exact snapshots of all original source files prior to edits.

---

## 4. Historical Result Comparability Assessment

The following historical benchmarks are **no longer directly comparable** and must be re-evaluated in Stage 3:

1. **Latent SMOTE (Historical Exp 4b):**
   - *Previous limitation:* Standalone `oversampling.py` extracted representations from an untrained MLP, while `run_controlled_benchmarks.py` used the class-weighted MLP.
   - *Status:* Corrected to extract embeddings from the genuinely trained unweighted baseline MLP. Future runs will reflect true latent-space SMOTE.
2. **Longitudinal Auxiliary Regression (Historical Multi-Task / Longitudinal):**
   - *Previous limitation:* The regression head `reg_score` was instantiated and evaluated but never had regression loss backpropagated during training.
   - *Status:* Corrected with combined loss ($L = L_{\text{cls}} + \lambda_{\text{reg}} L_{\text{reg}}$) and target masking.
3. **Multitask Architecture History Encoding:**
   - *Previous limitation:* Historical `multitask.py` allowed trailing padded zeros to affect GRU states.
   - *Status:* Corrected with length-indexed state extraction and `no_history_emb`.
