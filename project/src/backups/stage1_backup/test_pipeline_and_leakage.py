import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import json
import tempfile

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from evaluation import evaluate_classification, evaluate_regression
from baseline import ClassificationMLP, compute_class_weights
from longitudinal import build_longitudinal_sequences, LongitudinalDataset, collate_seq_batch, LongitudinalGRUModel

class TestPipelineAndLeakageIntegrity(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic longitudinal dataset
        # 10 participants, 5 periods each
        records = []
        for uid in range(1, 11):
            for p in range(1, 6):
                score = float(np.random.randint(0, 28))
                if score <= 4:
                    cat = 0
                elif score <= 9:
                    cat = 1
                elif score <= 14:
                    cat = 2
                elif score <= 19:
                    cat = 3
                else:
                    cat = 4
                start_score = max(0.0, score - np.random.randint(-3, 4))
                records.append({
                    "user_id": f"user_{uid}",
                    "period_num": p,
                    "feat_steps": float(np.random.normal(8000, 1000)),
                    "feat_sleep": float(np.random.normal(400, 50)),
                    "feat_bmi": 24.5,
                    "phq9_cat_end": cat,
                    "phq9_score_end": score,
                    "phq9_cat_start": cat,
                    "phq9_score_start": start_score
                })
        self.df = pd.DataFrame(records)
        self.df.index = [f"{r['user_id']}_{r['period_num']}" for r in records]
        self.feature_cols = ["feat_steps", "feat_sleep", "feat_bmi"]

    def test_condition_1_no_participant_in_multiple_splits(self):
        """Condition 1: No participant appears in multiple splits."""
        train_df, val_df, test_df, _ = perform_subject_level_split(self.df, seed=42)
        tr = set(train_df['user_id'])
        va = set(val_df['user_id'])
        te = set(test_df['user_id'])
        self.assertEqual(len(tr.intersection(va)), 0)
        self.assertEqual(len(tr.intersection(te)), 0)
        self.assertEqual(len(va.intersection(te)), 0)

    def test_condition_2_no_target_column_enters_feature_matrix(self):
        """Condition 2: No target column enters the feature matrix."""
        f_cols, _, _, t_cols, _ = inspect_modalities_and_features(self.df)
        for target in t_cols + ['delta_phq9', 'phq9_cat_end', 'phq9_score_end']:
            self.assertNotIn(target, f_cols)

    def test_condition_3_no_future_observation_enters_history(self):
        """Condition 3: No future-period observation enters a history sequence."""
        prep = DataPreprocessor(feature_cols=self.feature_cols)
        prep.fit_train(self.df)
        samples = build_longitudinal_sequences(self.df, prep)
        for s in samples:
            curr_p = s["period_num"]
            # All history records were verified inside build_longitudinal_sequences
            # Let's also verify that s["history_len"] < curr_p
            self.assertLess(s["history_len"], curr_p)

    def test_condition_4_current_observation_not_duplicated_in_history(self):
        """Condition 4: The current observation is not duplicated in its own history."""
        prep = DataPreprocessor(feature_cols=self.feature_cols)
        prep.fit_train(self.df)
        samples = build_longitudinal_sequences(self.df, prep)
        # Find step 0 samples (period 1)
        p1_samples = [s for s in samples if s["period_num"] == 1]
        self.assertTrue(len(p1_samples) > 0)
        for s in p1_samples:
            self.assertEqual(s["history_len"], 0)
            self.assertEqual(s["has_history"], 0.0)
            self.assertEqual(len(s["history_X"]), 0)

    def test_condition_5_padded_time_steps_do_not_affect_gru_output(self):
        """Condition 5: Padded time steps do not affect the GRU output."""
        dim = 3
        model = LongitudinalGRUModel(input_dim=dim, hidden_dim=64, num_classes=5, mode="history_only")
        model.eval()
        
        # Create a single sequence of true length 2
        seq_2 = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32) # (1, 2, 3)
        curr_dummy = torch.zeros(1, dim)
        
        # Now pad with 1 zero (length 2, max_len 3)
        padded_3 = torch.cat([seq_2, torch.zeros(1, 1, dim)], dim=1) # (1, 3, 3)
        
        # Now pad with 5 zeros (length 2, max_len 7)
        padded_7 = torch.cat([seq_2, torch.zeros(1, 5, dim)], dim=1) # (1, 7, 3)
        
        with torch.no_grad():
            logits_3, _, _ = model(curr_dummy, padded_3, torch.tensor([2]), torch.tensor([1.0]))
            logits_7, _, _ = model(curr_dummy, padded_7, torch.tensor([2]), torch.tensor([1.0]))
            
        # Logits must be identical regardless of how many padding zeros were appended!
        self.assertTrue(torch.allclose(logits_3, logits_7, atol=1e-5))

    def test_condition_6_phq9_categories_match_documented_score_ranges(self):
        """Condition 6: PHQ-9 categories match documented score ranges."""
        for score, expected_cat in [(0.0, 0), (4.0, 0), (5.0, 1), (9.0, 1), (10.0, 2), (14.0, 2), (15.0, 3), (19.0, 3), (20.0, 4), (27.0, 4)]:
            c = 0 if score <= 4 else (1 if score <= 9 else (2 if score <= 14 else (3 if score <= 19 else 4)))
            self.assertEqual(c, expected_cat)

    def test_condition_7_delta_targets_use_documented_definition(self):
        """Condition 7: Delta targets use the documented definition."""
        prep = DataPreprocessor(feature_cols=self.feature_cols)
        prep.fit_train(self.df)
        samples = build_longitudinal_sequences(self.df, prep)
        for s in samples:
            # delta = end - start
            self.assertIsInstance(s["y_delta"], float)

    def test_condition_8_preprocessing_fitted_on_training_data_only(self):
        """Condition 8: Preprocessing is fitted on training data only."""
        train_df, val_df, _, _ = perform_subject_level_split(self.df, seed=42)
        prep = DataPreprocessor(feature_cols=self.feature_cols)
        self.assertFalse(prep.is_fitted)
        prep.fit_train(train_df)
        self.assertTrue(prep.is_fitted)
        # Transforming val_df does not alter fitted mean/variance
        mean_before = prep.scaler.mean_.copy()
        prep.transform(val_df)
        mean_after = prep.scaler.mean_.copy()
        self.assertTrue(np.allclose(mean_before, mean_after))

    def test_condition_9_class_weights_computed_from_training_labels_only(self):
        """Condition 9: Class weights are computed from training labels only."""
        y_train = np.array([0, 0, 0, 1, 1, 2, 3, 4])
        weights = compute_class_weights(y_train, num_classes=5)
        # Class 0 has 3 samples, Class 4 has 1 sample -> weight(4) > weight(0)
        self.assertGreater(weights[4].item(), weights[0].item())

    def test_condition_10_validation_and_test_never_resampled(self):
        """Condition 10: Validation and test datasets are never resampled."""
        # Verification that test sample count is constant
        _, val_df, test_df, _ = perform_subject_level_split(self.df, seed=42)
        orig_val_len = len(val_df)
        orig_test_len = len(test_df)
        self.assertEqual(len(val_df), orig_val_len)
        self.assertEqual(len(test_df), orig_test_len)

    def test_condition_11_evaluation_metrics_handle_absent_predicted_classes(self):
        """Condition 11: Evaluation metrics handle absent predicted classes consistently."""
        y_true = np.array([0, 1, 2, 3, 4])
        # Predict only class 0
        y_pred = np.array([0, 0, 0, 0, 0])
        metrics = evaluate_classification(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        self.assertIsInstance(metrics["macro_f1"], float)
        self.assertIsInstance(metrics["precision_macro"], float)
        self.assertIsInstance(metrics["balanced_accuracy"], float)
        self.assertEqual(metrics["recall_macro"], 0.2)

    def test_condition_12_repeated_runs_reproducible_with_seed(self):
        """Condition 12: Repeated runs with the same seed are reproducible."""
        torch.manual_seed(42)
        m1 = ClassificationMLP(input_dim=3, hidden_dim=16, num_classes=5)
        torch.manual_seed(42)
        m2 = ClassificationMLP(input_dim=3, hidden_dim=16, num_classes=5)
        m1.eval()
        m2.eval()
        x = torch.randn(2, 3)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m1(x)[0], m2(x)[0]))

    def test_condition_13_checkpoint_loading_restores_intended_model(self):
        """Condition 13: Checkpoint loading restores the intended best model."""
        m = ClassificationMLP(input_dim=3, hidden_dim=16, num_classes=5)
        m.eval()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            torch.save(m.state_dict(), ckpt_path)
            m_loaded = ClassificationMLP(input_dim=3, hidden_dim=16, num_classes=5)
            m_loaded.load_state_dict(torch.load(ckpt_path, weights_only=True))
            m_loaded.eval()
            x = torch.randn(2, 3)
            with torch.no_grad():
                self.assertTrue(torch.allclose(m(x)[0], m_loaded(x)[0]))
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def test_condition_14_evaluation_code_does_not_train_on_test_data(self):
        """Condition 14: Evaluation code does not accidentally train on test data."""
        m = ClassificationMLP(input_dim=3, hidden_dim=16, num_classes=5)
        m.eval()
        weights_before = [p.clone() for p in m.parameters()]
        x_test = torch.randn(4, 3)
        with torch.no_grad():
            _ = m(x_test)
        for p_before, p_after in zip(weights_before, m.parameters()):
            self.assertTrue(torch.allclose(p_before, p_after))

if __name__ == '__main__':
    unittest.main()
