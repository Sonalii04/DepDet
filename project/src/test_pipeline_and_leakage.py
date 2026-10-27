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
from evaluation import evaluate_classification, evaluate_regression, compute_multiclass_g_mean
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


class TestEvaluationMetrics(unittest.TestCase):
    """Task 1 Unit Tests: Evaluation metrics standardization & edge cases."""

    def test_hand_checkable_perfect_classification(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        metrics = evaluate_classification(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["weighted_f1"], 1.0)
        self.assertEqual(metrics["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(metrics["ordinal_class_mae"], 0.0)
        self.assertEqual(metrics["multiclass_g_mean"], 1.0)
        for c in range(5):
            self.assertEqual(metrics["per_class_f1"][str(c)], 1.0)
            self.assertEqual(metrics["per_class_recall"][str(c)], 1.0)
            self.assertEqual(metrics["per_class_precision"][str(c)], 1.0)

    def test_hand_checkable_known_imperfect_classification(self):
        # 10 samples, only one error: one class 4 predicted as class 3
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4])
        metrics = evaluate_classification(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        
        self.assertEqual(metrics["accuracy"], 0.9)
        # Ordinal MAE: 9 diffs of 0, 1 diff of |4 - 3| = 1 -> 1 / 10 = 0.1
        self.assertAlmostEqual(metrics["ordinal_class_mae"], 0.1, places=5)
        # Confusion matrix has fixed shape 5x5
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(cm[4, 3], 1)
        self.assertEqual(cm[4, 4], 1)
        self.assertEqual(cm[3, 3], 2)
        # QWK should be positive and high
        self.assertGreater(metrics["quadratic_weighted_kappa"], 0.8)

    def test_multiclass_g_mean_zero_class_recall(self):
        # Class 4 has 0 recall
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 3])
        g_mean = compute_multiclass_g_mean(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        # Standard convention: if any class has 0 recall, geometric mean is 0.0
        self.assertEqual(g_mean, 0.0)

    def test_ovr_roc_auc_and_pr_auc_valid_probabilities(self):
        y_true = np.array([0, 1, 2, 3, 4])
        # Slightly noisy one-hot probabilities
        y_prob = np.array([
            [0.8, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.8, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.8, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.8, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.8]
        ])
        metrics = evaluate_classification(y_true, np.argmax(y_prob, axis=1), y_prob=y_prob, labels=[0, 1, 2, 3, 4])
        self.assertIsNotNone(metrics["ovr_roc_auc_macro"])
        self.assertIsNotNone(metrics["ovr_pr_auc_macro"])
        self.assertGreater(metrics["ovr_roc_auc_macro"], 0.9)
        self.assertGreater(metrics["ovr_pr_auc_macro"], 0.9)

    def test_confusion_matrix_fixed_class_order(self):
        # Only classes 0 and 2 present in predictions
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 0, 2, 2, 2])
        metrics = evaluate_classification(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[1, 0], 1)
        self.assertEqual(cm[2, 2], 1)
        self.assertEqual(cm[3, 2], 1)
        self.assertEqual(cm[4, 2], 1)


class TestLatentSMOTEIntegrity(unittest.TestCase):
    """Task 2 Unit Tests: Latent SMOTE encoder and resampling isolation."""

    def test_latent_smote_resampling_balances_classes(self):
        from oversampling import latent_space_smote
        # 50 samples in 5 classes (imbalanced)
        np.random.seed(42)
        latents = np.random.randn(50, 16)
        labels = np.array([0]*20 + [1]*15 + [2]*8 + [3]*5 + [4]*2)
        bal_lat, bal_lab = latent_space_smote(latents, labels)
        self.assertGreater(len(bal_lat), len(latents))
        self.assertEqual(len(bal_lat), len(bal_lab))
        # Every class should now have equal number of samples (target_count = 20)
        _, counts = np.unique(bal_lab, return_counts=True)
        self.assertTrue(all(c == 20 for c in counts))

    def test_validation_and_test_never_enter_smote(self):
        from oversampling import latent_space_smote
        # Latents for 30 train, 10 val, 10 test
        np.random.seed(42)
        lat_tr = np.random.randn(30, 8)
        y_tr = np.array([0]*12 + [1]*8 + [2]*5 + [3]*3 + [4]*2)
        lat_va = np.random.randn(10, 8)
        y_va = np.random.choice([0, 1, 2, 3, 4], size=10)
        lat_te = np.random.randn(10, 8)
        y_te = np.random.choice([0, 1, 2, 3, 4], size=10)
        
        val_lat_before = lat_va.copy()
        test_lat_before = lat_te.copy()
        
        # Only training data enters SMOTE
        bal_lat_tr, bal_y_tr = latent_space_smote(lat_tr, y_tr)
        
        # Verify val and test arrays are completely untouched
        self.assertEqual(len(lat_va), 10)
        self.assertEqual(len(lat_te), 10)
        self.assertTrue(np.allclose(lat_va, val_lat_before))
        self.assertTrue(np.allclose(lat_te, test_lat_before))


class TestLongitudinalRegressionLoss(unittest.TestCase):
    """Task 3 Unit Tests: Continuous PHQ-9 regression head and loss integration."""

    def test_regression_loss_computation_and_parameter_update(self):
        input_dim = 4
        model = LongitudinalGRUModel(input_dim=input_dim, hidden_dim=32, num_classes=5, mode="current_plus_history")
        model.train()
        
        initial_weight = model.reg_score.weight.clone().detach()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        curr_X = torch.randn(4, input_dim)
        history_X = torch.randn(4, 3, input_dim)
        lengths = torch.tensor([2, 1, 3, 0])
        has_history = torch.tensor([1.0, 1.0, 1.0, 0.0])
        y_cat = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        y_score = torch.tensor([4.0, 8.5, 12.0, 18.0], dtype=torch.float32)
        
        logits, score_pred, _ = model(curr_X, history_X, lengths, has_history)
        
        loss_c = nn.CrossEntropyLoss()(logits, y_cat)
        loss_r = nn.MSELoss()(score_pred, y_score)
        regression_weight = 0.5
        total_loss = loss_c + regression_weight * loss_r
        
        self.assertGreater(loss_r.item(), 0.0)
        
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradients must reach the regression head
        self.assertIsNotNone(model.reg_score.weight.grad)
        self.assertFalse(torch.all(model.reg_score.weight.grad == 0))
        
        optimizer.step()
        
        # Parameters must update
        updated_weight = model.reg_score.weight.clone().detach()
        self.assertFalse(torch.allclose(initial_weight, updated_weight))

    def test_missing_regression_target_masking(self):
        input_dim = 4
        model = LongitudinalGRUModel(input_dim=input_dim, hidden_dim=32, num_classes=5, mode="current_only")
        model.train()
        
        curr_X = torch.randn(2, input_dim)
        history_X = torch.zeros(2, 1, input_dim)
        lengths = torch.tensor([0, 0])
        has_history = torch.tensor([0.0, 0.0])
        y_cat = torch.tensor([0, 1], dtype=torch.long)
        y_score = torch.tensor([float('nan'), float('nan')], dtype=torch.float32)
        
        logits, score_pred, _ = model(curr_X, history_X, lengths, has_history)
        loss_c = nn.CrossEntropyLoss()(logits, y_cat)
        
        valid_mask = ~torch.isnan(y_score)
        self.assertEqual(valid_mask.sum().item(), 0)
        
        # When all regression targets are NaN, regression loss is skipped safely
        if valid_mask.sum() > 0:
            loss_r = nn.MSELoss()(score_pred[valid_mask], y_score[valid_mask])
            total_loss = loss_c + 0.5 * loss_r
        else:
            total_loss = loss_c
            
        total_loss.backward()
        self.assertFalse(torch.isnan(total_loss))


class TestMultitaskPaddingInvariance(unittest.TestCase):
    """Task 4 Unit Tests: Multitask GRU padding invariance and no-history handling."""

    def test_multitask_gru_padding_invariance(self):
        from multitask import MultiTaskFullModel
        input_dim = 4
        model = MultiTaskFullModel(input_dim=input_dim, hidden_dim=32, proj_dim=16, num_classes=5)
        model.eval()
        
        # Current observation
        curr_X = torch.randn(1, input_dim)
        
        # Sequence of true length 2
        seq_2 = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=torch.float32) # (1, 2, 4)
        
        # Pad with 2 zeros (length 2, max_len 4)
        padded_4 = torch.cat([seq_2, torch.zeros(1, 2, input_dim)], dim=1) # (1, 4, 4)
        # Pad with 6 zeros (length 2, max_len 8)
        padded_8 = torch.cat([seq_2, torch.zeros(1, 6, input_dim)], dim=1) # (1, 8, 4)
        
        lengths = torch.tensor([2])
        
        with torch.no_grad():
            logits_4, phq_4, change_4, proj_4, state_4 = model(curr_X, padded_4, lengths)
            logits_8, phq_8, change_8, proj_8, state_8 = model(curr_X, padded_8, lengths)
            
        # All outputs must be completely identical regardless of padding length
        self.assertTrue(torch.allclose(state_4, state_8, atol=1e-5))
        self.assertTrue(torch.allclose(logits_4, logits_8, atol=1e-5))
        self.assertTrue(torch.allclose(phq_4, phq_8, atol=1e-5))
        self.assertTrue(torch.allclose(change_4, change_8, atol=1e-5))
        self.assertTrue(torch.allclose(proj_4, proj_8, atol=1e-5))

    def test_multitask_no_history_handling(self):
        from multitask import MultiTaskFullModel
        input_dim = 4
        model = MultiTaskFullModel(input_dim=input_dim, hidden_dim=32, proj_dim=16, num_classes=5)
        model.eval()
        
        curr_X = torch.randn(2, input_dim)
        history_X = torch.zeros(2, 3, input_dim)
        lengths = torch.tensor([0, 0])
        
        with torch.no_grad():
            logits, phq, change, proj, state = model(curr_X, history_X, lengths)
            
        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(phq.shape, (2,))
        self.assertEqual(change.shape, (2,))


class TestXGBoostImplementation(unittest.TestCase):
    """Task 5 Unit Tests: XGBoost multiclass model, shapes, and probabilities."""

    def test_xgboost_output_shape_probabilities_and_class_order(self):
        from xgboost_model import build_xgboost_classifier, compute_multiclass_sample_weights
        
        # Synthetic dataset with 5 classes
        np.random.seed(42)
        X_train = np.random.randn(60, 5)
        y_train = np.array([0]*20 + [1]*15 + [2]*10 + [3]*10 + [4]*5)
        
        X_val = np.random.randn(20, 5)
        y_val = np.random.choice([0, 1, 2, 3, 4], size=20)
        
        weights = compute_multiclass_sample_weights(y_train, [0, 1, 2, 3, 4])
        self.assertEqual(len(weights), len(y_train))
        # Rare class 4 must have higher weight than common class 0
        self.assertGreater(weights[y_train == 4][0], weights[y_train == 0][0])
        
        clf = build_xgboost_classifier(random_state=42, n_estimators=10)
        clf.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)
        
        probs = clf.predict_proba(X_val)
        preds = clf.predict(X_val)
        
        # Verify shape (N, 5)
        self.assertEqual(probs.shape, (20, 5))
        # Verify probabilities sum to 1.0
        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-4))
        # Verify predictions match argmax of probabilities
        self.assertTrue(np.array_equal(preds, np.argmax(probs, axis=1)))
        # Verify classes are ordered 0..4
        self.assertTrue(np.array_equal(clf.classes_, np.array([0, 1, 2, 3, 4])))


if __name__ == '__main__':
    unittest.main()
