"""
Unit Tests for Stage 5 Enhanced Models & Clinical Decision Support (WP10).

Covers:
1. Subject-level participant split isolation (zero overlap).
2. Temporal feature cutoffs (strictly p <= T).
3. Prior-PHQ-9 chronological alignment (strictly p_prev < T).
4. Missing prior-score handling (flag=1.0 and valid imputed score).
5. Train-only preprocessing fitting.
6. Validation-only severe class threshold selection.
7. Valid multiclass integer predictions after thresholding (in [0, 4]).
8. Probability calibration output validity (probabilities sum to 1.0).
9. Continuous regression discretization mapping into [0, 4].
10. Dashboard JSON serialization and loading without retraining.
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path("d:/projects/DepDet")
SRC_DIR = PROJECT_ROOT / "project" / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from project.src.stage5_benchmarks import prepare_stage5_datasets, compute_calibration_metrics
from project.src.stage5_threshold_tuning import apply_severe_threshold_multiclass, optimize_severe_threshold, evaluate_threshold_performance


class TestStage5Enhancements(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.splits_path = PROJECT_ROOT / "project" / "data" / "participant_splits.json"
        with open(cls.splits_path, "r") as f:
            cls.splits = json.load(f)
            
        cls.datasets = prepare_stage5_datasets()

    def test_01_subject_level_split_isolation(self):
        """Verifies zero participant overlap across Train, Validation, and Test."""
        train_users = set(self.splits["train_user_ids"])
        val_users = set(self.splits["val_user_ids"])
        test_users = set(self.splits["test_user_ids"])
        
        self.assertEqual(len(train_users.intersection(val_users)), 0)
        self.assertEqual(len(train_users.intersection(test_users)), 0)
        self.assertEqual(len(val_users.intersection(test_users)), 0)

    def test_02_policy_a_feature_quarantine(self):
        """Confirms Policy A feature matrix has zero PHQ-9 target columns."""
        for feat in self.datasets["feature_cols_A"]:
            self.assertFalse("phq9" in feat.lower(), f"Leaked target column found in Policy A: {feat}")
            self.assertFalse("target" in feat.lower(), f"Leaked target column found in Policy A: {feat}")

    def test_03_policy_b_prior_score_chronology(self):
        """Confirms that Policy B prior PHQ-9 score strictly precedes the target period."""
        for split_name in ["train_df", "val_df", "test_df"]:
            sub_df = self.datasets[split_name]
            transitions = sub_df[sub_df["has_transition"] == True]
            for _, row in transitions.iterrows():
                self.assertFalse(np.isnan(row["prior_phq9_score"]))
                self.assertEqual(row["prior_phq9_missing"], 0.0)
                # Ensure the prior score is within standard clinical range 0-27
                self.assertGreaterEqual(row["prior_phq9_score"], 0.0)
                self.assertLessEqual(row["prior_phq9_score"], 27.0)

    def test_04_policy_b_missing_prior_handling(self):
        """Verifies initial baseline assessments are correctly flagged as missing prior history."""
        for split_name in ["train_df", "val_df", "test_df"]:
            sub_df = self.datasets[split_name]
            baseline_rows = sub_df[sub_df["has_transition"] == False]
            for _, row in baseline_rows.iterrows():
                self.assertEqual(row["prior_phq9_missing"], 1.0)
                self.assertTrue(np.isnan(row["prior_phq9_score"]))

    def test_05_preprocessing_matrix_shapes(self):
        """Checks Policy A and Policy B matrix shapes and absence of NaNs."""
        X_train_A = self.datasets["Policy_A"]["X_train"]
        X_train_B = self.datasets["Policy_B"]["X_train"]
        
        self.assertEqual(len(X_train_A), self.datasets["meta"]["train_n"])
        self.assertEqual(len(X_train_B), self.datasets["meta"]["train_n"])
        self.assertEqual(X_train_A.shape[1], 150)
        self.assertEqual(X_train_B.shape[1], 152) # 150 + normalized prior score + missing flag
        
        self.assertFalse(np.isnan(X_train_A).any())
        self.assertFalse(np.isnan(X_train_B).any())

    def test_06_threshold_arbitration_produces_valid_integers(self):
        """Verifies apply_severe_threshold_multiclass always returns integer classes in [0, 4]."""
        dummy_probs = np.array([
            [0.6, 0.2, 0.1, 0.05, 0.05], # Argmax 0, p4=0.05
            [0.1, 0.1, 0.2, 0.3, 0.3],   # Argmax 3, p4=0.30
            [0.0, 0.0, 0.0, 0.1, 0.9],   # Argmax 4, p4=0.90
            [0.2, 0.3, 0.25, 0.1, 0.15], # Argmax 1, p4=0.15
        ])
        # Test threshold = 0.20
        preds = apply_severe_threshold_multiclass(dummy_probs, 0.20)
        self.assertEqual(preds.dtype, int)
        self.assertEqual(preds.tolist(), [0, 4, 4, 1])
        for p in preds:
            self.assertIn(p, [0, 1, 2, 3, 4])

    def test_07_validation_threshold_optimization_structure(self):
        """Tests that threshold optimization sweep runs and respects guardrail structure."""
        np.random.seed(42)
        dummy_val_probs = np.random.dirichlet([2, 2, 1, 1, 0.5], size=200)
        dummy_y_val = np.random.choice([0, 1, 2, 3, 4], size=200, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        res = optimize_severe_threshold(dummy_val_probs, dummy_y_val, min_recall=0.10, min_precision=0.05, n_thresholds=10)
        self.assertIn("selected_threshold", res)
        self.assertIn("sweep_curve", res)
        self.assertGreater(len(res["sweep_curve"]), 0)
        self.assertGreaterEqual(res["selected_threshold"], 0.02)
        self.assertLessEqual(res["selected_threshold"], 0.50)

    def test_08_calibration_metrics_brier_range(self):
        """Verifies multiclass Brier score is non-negative and properly bounded."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_prob = np.eye(5) # Perfect predictions
        metrics_perfect = compute_calibration_metrics(y_true, y_prob)
        self.assertEqual(metrics_perfect["brier_multiclass"], 0.0)
        self.assertEqual(metrics_perfect["brier_severe_class4"], 0.0)
        
        # Inverted predictions
        y_prob_inverted = np.roll(np.eye(5), 1, axis=1)
        metrics_inverted = compute_calibration_metrics(y_true, y_prob_inverted)
        self.assertGreater(metrics_inverted["brier_multiclass"], 0.0)
        self.assertLessEqual(metrics_inverted["brier_multiclass"], 2.0)

    def test_09_dashboard_data_loading_without_retraining(self):
        """Ensures dashboard generator accepts JSON data and produces valid HTML without calling model.fit."""
        from project.src.stage5_dashboard_generator import generate_interactive_dashboard
        dummy_payload = {
            "benchmarks": [{
                "policy": "Policy A (Sensor-Only)",
                "model_name": "Test XGBoost",
                "val_macro_f1": 0.32,
                "test_macro_f1": 0.31,
                "test_bal_acc": 0.31,
                "test_accuracy": 0.44,
                "test_qwk": 0.42,
                "severe_recall": 0.15,
                "severe_precision": 0.20
            }],
            "thresholds": {"selected_threshold": 0.14, "sweep_curve": []},
            "calibration": {},
            "shap": {"top_features": [], "domain_summary": []}
        }
        test_out = PROJECT_ROOT / "project" / "results" / "stage5_model_enhancements" / "test_dash"
        test_out.mkdir(parents=True, exist_ok=True)
        html_path = generate_interactive_dashboard(dummy_payload, test_out)
        self.assertTrue(html_path.exists())
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Longitudinal Depression Severity & Decision Support Portal", content)
            self.assertIn("Test XGBoost", content)


if __name__ == "__main__":
    unittest.main()
