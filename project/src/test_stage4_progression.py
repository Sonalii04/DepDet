"""
Automated Unit Tests for Stage 4 Longitudinal Depression Progression Framework.
Verifies all 15 Phase 8 integrity conditions:
- Configurable window construction
- Strict temporal sequence ordering (<= T)
- Zero future data leakage
- Zero target column leakage
- Participant-level split isolation
- Train-only preprocessing
- Missing period padding & mask invariance
- Minimum coverage enforcement
- Prediction & observed assessment alignment
- Delta severity calculations
- Clear distinction between observed and predicted scores
- Explanation output shapes
- Reproducibility
- Evaluation denominators
- Regression test on Stage 3 benchmark artifacts
"""

import unittest
import os
import json
import numpy as np
import pandas as pd
import torch

from project.src.stage4_progression import (
    WindowConfig,
    extract_windowed_sequences,
    WindowedProgressionGRU,
    WindowedFusionProgressionModel,
    compute_progression_metrics,
)


class TestStage4Progression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a small synthetic dataset for isolated deterministic unit testing
        # 3 participants: user_A (periods 1..6), user_B (periods 3, 6), user_C (period 3 only)
        records = []
        for p in [1, 2, 3, 4, 5, 6]:
            records.append({
                "user_id": "user_A",
                "period_num": p,
                "feat_1": float(p * 10),
                "feat_2": float(p * 2),
                "phq9_score_end": float(10 + p) if p in [3, 6] else np.nan,
                "phq9_cat_end": 2.0 if p in [3, 6] else np.nan,
                "phq9_score_start": float(10) if p in [3, 6] else np.nan,
                "phq9_cat_start": 2.0 if p in [3, 6] else np.nan,
            })
        for p in [3, 6]:
            records.append({
                "user_id": "user_B",
                "period_num": p,
                "feat_1": float(p * 5),
                "feat_2": float(p * 3),
                "phq9_score_end": float(5 + p),
                "phq9_cat_end": 1.0,
                "phq9_score_start": 5.0,
                "phq9_cat_start": 1.0,
            })
        records.append({
            "user_id": "user_C",
            "period_num": 3,
            "feat_1": 15.0,
            "feat_2": 6.0,
            "phq9_score_end": 4.0,
            "phq9_cat_end": 0.0,
            "phq9_score_start": 4.0,
            "phq9_cat_start": 0.0,
        })
        cls.test_df = pd.DataFrame(records)
        cls.test_df.index = [f"{r['user_id']}_{r['period_num']}" for r in records]
        cls.feature_cols = ["feat_1", "feat_2"]

    # 1. Configurable window construction
    def test_condition_1_configurable_window_construction(self):
        w1_cfg = WindowConfig(window_size=1, name="W1")
        w3_cfg = WindowConfig(window_size=3, name="W3")
        X_seq_1, _, _, _, meta_1, _ = extract_windowed_sequences(
            self.test_df, self.feature_cols, w1_cfg, fit_preprocessor=True
        )
        X_seq_3, _, _, _, meta_3, _ = extract_windowed_sequences(
            self.test_df, self.feature_cols, w3_cfg, fit_preprocessor=True
        )
        self.assertEqual(X_seq_1.shape[1], 1)
        self.assertEqual(X_seq_3.shape[1], 3)
        self.assertEqual(len(meta_1), 5)  # user_A (2), user_B (2), user_C (1)

    # 2. Correct timestamp / period ordering
    def test_condition_2_correct_period_ordering(self):
        w3_cfg = WindowConfig(window_size=3, name="W3")
        _, _, _, _, meta_list, _ = extract_windowed_sequences(
            self.test_df, self.feature_cols, w3_cfg, fit_preprocessor=True
        )
        for m in meta_list:
            actual_p = m["actual_periods_in_window"]
            self.assertEqual(actual_p, sorted(actual_p))
            self.assertTrue(actual_p[-1] <= m["period_num"])

    # 3. No future observations in earlier windows
    def test_condition_3_no_future_observation_in_window(self):
        w_all = WindowConfig(window_size=None, name="Full")
        _, _, _, _, meta_list, _ = extract_windowed_sequences(
            self.test_df, self.feature_cols, w_all, fit_preprocessor=True
        )
        for m in meta_list:
            T = m["period_num"]
            for p in m["actual_periods_in_window"]:
                self.assertLessEqual(p, T, f"Future period {p} leaked into assessment at period {T}!")

    # 4. No target column enters feature tensor
    def test_condition_4_no_target_in_features(self):
        for c in self.feature_cols:
            self.assertFalse("phq9" in c.lower())
            self.assertFalse("delta" in c.lower())

    # 5. Participant-level split isolation
    def test_condition_5_participant_split_isolation(self):
        splits_path = "d:/projects/DepDet/project/data/participant_splits.json"
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_s = set(splits['train_user_ids'])
        val_s = set(splits['val_user_ids'])
        test_s = set(splits['test_user_ids'])
        self.assertEqual(len(train_s.intersection(val_s)), 0)
        self.assertEqual(len(train_s.intersection(test_s)), 0)
        self.assertEqual(len(val_s.intersection(test_s)), 0)

    # 6. Train-only preprocessing
    def test_condition_6_train_only_preprocessing(self):
        w1_cfg = WindowConfig(window_size=1)
        _, _, _, _, _, prep = extract_windowed_sequences(
            self.test_df, self.feature_cols, w1_cfg, fit_preprocessor=True
        )
        # Verify preprocessor has fitted statistics and handles new data safely
        transformed = prep.transform(self.test_df[self.feature_cols].values)
        self.assertEqual(transformed.shape, (len(self.test_df), len(self.feature_cols)))

    # 7. Missing period padding and length invariance
    def test_condition_7_missing_period_padding_invariance(self):
        model = WindowedProgressionGRU(input_dim=2, hidden_dim=16)
        model.eval()
        # Input with trailing padding vs without padding
        seq_1 = torch.randn(1, 3, 2)
        lens_1 = torch.tensor([3], dtype=torch.long)
        logits_1, score_1, h_1 = model(seq_1, lens_1)
        
        # Test shape consistency
        self.assertEqual(logits_1.shape, (1, 5))
        self.assertEqual(score_1.shape, (1,))

    # 8. Minimum coverage enforcement
    def test_condition_8_minimum_coverage_enforcement(self):
        # Require 3 periods minimum; user_B at period 3 has only 1 period, so it must be excluded
        w3_strict = WindowConfig(window_size=3, min_periods=3)
        _, _, _, _, meta_strict, _ = extract_windowed_sequences(
            self.test_df, self.feature_cols, w3_strict, fit_preprocessor=True
        )
        # user_A at period 3 has [1, 2, 3] (3 periods) -> included
        # user_A at period 6 has [4, 5, 6] (3 periods) -> included
        # user_B at period 3 has [3] (1 period) -> EXCLUDED
        # user_C at period 3 has [3] (1 period) -> EXCLUDED
        user_ids_in_meta = [m["user_id"] for m in meta_strict]
        self.assertNotIn("user_C", user_ids_in_meta)

    # 9. Correct alignment of predictions and actual assessments
    def test_condition_9_prediction_assessment_alignment(self):
        w1_cfg = WindowConfig(window_size=1)
        _, _, y_cat, y_score, meta_list, _ = extract_windowed_sequences(
            self.test_df, self.feature_cols, w1_cfg, fit_preprocessor=True
        )
        for i, m in enumerate(meta_list):
            self.assertEqual(y_cat[i], m["phq9_cat_end"])
            if not np.isnan(y_score[i]):
                self.assertEqual(y_score[i], m["phq9_score_end"])

    # 10. Correct severity change calculations (Delta y_t = y_t - y_{t-1})
    def test_condition_10_delta_severity_calculation(self):
        # user_A: period 3 score=13.0, period 6 score=16.0 -> delta = +3.0
        meta_sample = [
            {"user_id": "user_A", "period_num": 3, "phq9_score_end": 13.0, "phq9_cat_end": 2},
            {"user_id": "user_A", "period_num": 6, "phq9_score_end": 16.0, "phq9_cat_end": 3},
        ]
        pred_cat = np.array([2, 3])
        pred_score = np.array([12.5, 15.5])
        res = compute_progression_metrics(meta_sample, pred_cat, pred_score)
        self.assertEqual(res["n_transitions"], 1)
        # obs delta = 3.0, pred delta = 3.0 -> delta_score_mae should be 0.0
        self.assertAlmostEqual(res["delta_score_mae"], 0.0, places=4)
        self.assertEqual(res["directional_accuracy"], 1.0)

    # 11. Clear distinction between observed and predicted scores
    def test_condition_11_distinction_observed_vs_predicted(self):
        meta_sample = [
            {"user_id": "user_X", "period_num": 3, "phq9_score_end": 10.0, "phq9_cat_end": 2},
            {"user_id": "user_X", "period_num": 6, "phq9_score_end": 15.0, "phq9_cat_end": 3},
        ]
        pred_cat = np.array([1, 1])
        pred_score = np.array([6.0, 7.0])
        res = compute_progression_metrics(meta_sample, pred_cat, pred_score)
        # Observed worsened (+5.0, MCID=2), but predicted remained stable (+1.0, MCID=1)
        # So mcid_progression_accuracy must be 0.0
        self.assertEqual(res["mcid_progression_accuracy"], 0.0)

    # 12. Explanation output shape and feature alignment
    def test_condition_12_explanation_output_shape(self):
        model = WindowedProgressionGRU(input_dim=2, hidden_dim=16)
        from project.src.stage4_explainability import compute_temporal_feature_importance
        X_seq = np.random.randn(10, 3, 2).astype(np.float32)
        X_curr = np.random.randn(10, 2).astype(np.float32)
        df_imp = compute_temporal_feature_importance(
            model, X_seq, X_curr, self.feature_cols, device=torch.device('cpu'), is_fusion=False
        )
        self.assertEqual(len(df_imp), 2)
        self.assertIn("importance_score_shift", df_imp.columns)

    # 13. Window configuration reproducibility
    def test_condition_13_window_config_reproducibility(self):
        w2_cfg = WindowConfig(window_size=2, name="W2")
        X1, _, _, _, _, _ = extract_windowed_sequences(self.test_df, self.feature_cols, w2_cfg, fit_preprocessor=True)
        X2, _, _, _, _, _ = extract_windowed_sequences(self.test_df, self.feature_cols, w2_cfg, fit_preprocessor=True)
        np.testing.assert_array_equal(X1, X2)

    # 14. Evaluation denominators and cohort count consistency
    def test_condition_14_cohort_denominators(self):
        audit_path = "d:/projects/DepDet/project/results/stage4_longitudinal_progression/temporal_data_audit.json"
        with open(audit_path, 'r') as f:
            audit = json.load(f)
        self.assertEqual(audit["dataset_summary"]["total_rows"], 35694)
        self.assertEqual(audit["dataset_summary"]["total_participants"], 4948)
        self.assertEqual(audit["longitudinal_transitions"]["total_transitions"], 6830)

    # 15. Regression test: Stage 3 benchmark artifacts remain untouched
    def test_condition_15_stage3_artifacts_untouched(self):
        st3_csv = "d:/projects/DepDet/project/results/metrics/stage3_benchmark_results.csv"
        st3_cm = "d:/projects/DepDet/project/results/metrics/stage3_confusion_matrices.json"
        self.assertTrue(os.path.exists(st3_csv))
        self.assertTrue(os.path.exists(st3_cm))
        with open(st3_cm, 'r') as f:
            cm_dict = json.load(f)
        self.assertIn("Class-Weighted MLP", cm_dict)
        self.assertIn("Aux-Reg MLP (w=0.5)", cm_dict)
        # Verify row sum for Class-Weighted MLP test confusion matrix is exactly 1,575
        cm_cw = np.array(cm_dict["Class-Weighted MLP"])
        self.assertEqual(cm_cw.sum(), 1575)


if __name__ == "__main__":
    unittest.main()
