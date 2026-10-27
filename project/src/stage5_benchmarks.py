"""
Stage 5: Benchmarks, Dual Clinical Policies, Calibration, and Score Regression.

Implements:
1. Data preparation for Policy A (Sensor-Only) and Policy B (Prior-PHQ-9-Assisted).
2. XGBoost and HistGradientBoosting classifiers with class weighting.
3. Probability calibration (Platt Sigmoid and Isotonic Regression) with Brier score.
4. Continuous PHQ-9 score regression with clinical tier discretization.
5. Standardized evaluation on Validation and Held-Out Test sets.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, root_mean_squared_error
import xgboost as xgb

PROJECT_ROOT = Path("d:/projects/DepDet")
SRC_DIR = PROJECT_ROOT / "project" / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from project.src.data_loader import inspect_modalities_and_features
from project.src.evaluation import evaluate_classification, compute_multiclass_g_mean
from project.src.xgboost_model import compute_multiclass_sample_weights


def prepare_stage5_datasets():
    """
    Loads raw parquet and participant splits, then builds:
    - Policy A: Sensor-Only features (150 features, no prior or current PHQ-9).
    - Policy B: Prior-PHQ-9-Assisted features (150 features + prior_phq9_score + prior_phq9_missing).
    
    Guarantees:
    - Strictly chronological prior score extraction (period < T).
    - Missing prior scores handled with an explicit indicator flag and train-derived median imputation.
    - Zero participant overlap across splits.
    """
    parquet_path = PROJECT_ROOT / "anon_processed_df_parquet"
    splits_path = PROJECT_ROOT / "project" / "data" / "participant_splits.json"
    
    df = pd.read_parquet(parquet_path)
    idx_parts = pd.Series(df.index.astype(str)).str.rsplit('_', n=1, expand=True)
    df['user_id'] = idx_parts[0].values
    df['period_num'] = idx_parts[1].astype(int).values
    
    feature_cols, _, _, _, _ = inspect_modalities_and_features(df)
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    train_users = set(splits['train_user_ids'])
    val_users = set(splits['val_user_ids'])
    test_users = set(splits['test_user_ids'])
    
    # Sort strictly by participant and period
    df_sorted = df.sort_values(['user_id', 'period_num']).copy()
    
    # Pre-extract prior PHQ-9 survey score per user strictly from earlier periods
    # Survey rows have valid phq9_cat_end
    survey_mask = df_sorted['phq9_cat_end'].notna()
    
    user_surveys = {}
    for _, row in df_sorted[survey_mask].iterrows():
        u = row['user_id']
        p = int(row['period_num'])
        score = float(row['phq9_score_end']) if pd.notna(row['phq9_score_end']) else np.nan
        cat = int(row['phq9_cat_end'])
        if u not in user_surveys:
            user_surveys[u] = []
        user_surveys[u].append((p, score, cat))
        
    # Annotate every survey row with its latest antecedent survey
    prior_scores = []
    prior_cats = []
    prior_missing = []
    has_transition_flag = []
    
    df_labeled = df_sorted[survey_mask].copy()
    
    for _, row in df_labeled.iterrows():
        u = row['user_id']
        T = int(row['period_num'])
        
        # Look for the latest survey from this user with period < T
        earlier = [item for item in user_surveys.get(u, []) if item[0] < T]
        if len(earlier) > 0:
            latest = earlier[-1]
            prior_scores.append(latest[1])
            prior_cats.append(latest[2])
            prior_missing.append(0.0)
            has_transition_flag.append(True)
        else:
            prior_scores.append(np.nan)
            prior_cats.append(-1)
            prior_missing.append(1.0)
            has_transition_flag.append(False)
            
    df_labeled['prior_phq9_score'] = prior_scores
    df_labeled['prior_phq9_cat'] = prior_cats
    df_labeled['prior_phq9_missing'] = prior_missing
    df_labeled['has_transition'] = has_transition_flag
    
    # Separate into splits
    train_mask = df_labeled['user_id'].isin(train_users)
    val_mask = df_labeled['user_id'].isin(val_users)
    test_mask = df_labeled['user_id'].isin(test_users)
    
    train_df = df_labeled[train_mask].copy()
    val_df = df_labeled[val_mask].copy()
    test_df = df_labeled[test_mask].copy()
    
    # Targets
    y_train = train_df['phq9_cat_end'].astype(int).values
    y_val = val_df['phq9_cat_end'].astype(int).values
    y_test = test_df['phq9_cat_end'].astype(int).values
    
    y_score_train = train_df['phq9_score_end'].astype(float).values
    y_score_val = val_df['phq9_score_end'].astype(float).values
    y_score_test = test_df['phq9_score_end'].astype(float).values
    
    # Policy A: 150 Sensor/Demographic/Clinical features only
    X_train_raw_A = train_df[feature_cols].copy()
    X_val_raw_A = val_df[feature_cols].copy()
    X_test_raw_A = test_df[feature_cols].copy()
    
    # Fit Imputer and Scaler exclusively on Train
    imputer_A = SimpleImputer(strategy='median')
    scaler_A = StandardScaler()
    
    X_train_A = scaler_A.fit_transform(imputer_A.fit_transform(X_train_raw_A))
    X_val_A = scaler_A.transform(imputer_A.transform(X_val_raw_A))
    X_test_A = scaler_A.transform(imputer_A.transform(X_test_raw_A))
    
    # Policy B: Sensor features + Prior PHQ-9 (score + missing flag)
    # Impute prior_phq9_score using train-only median
    train_prior_median = float(train_df['prior_phq9_score'].dropna().median())
    
    def format_policy_b(sub_df, X_scaled):
        p_score = sub_df['prior_phq9_score'].fillna(train_prior_median).values.reshape(-1, 1)
        p_miss = sub_df['prior_phq9_missing'].values.reshape(-1, 1)
        # Normalize prior score using approximate range 0-27
        p_score_norm = (p_score - 10.0) / 6.0
        return np.hstack([X_scaled, p_score_norm, p_miss])
        
    X_train_B = format_policy_b(train_df, X_train_A)
    X_val_B = format_policy_b(val_df, X_val_A)
    X_test_B = format_policy_b(test_df, X_test_A)
    
    datasets = {
        "feature_cols_A": feature_cols,
        "feature_cols_B": feature_cols + ['prior_phq9_score_norm', 'prior_phq9_missing'],
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "Policy_A": {
            "X_train": X_train_A,
            "y_train": y_train,
            "y_score_train": y_score_train,
            "X_val": X_val_A,
            "y_val": y_val,
            "y_score_val": y_score_val,
            "X_test": X_test_A,
            "y_test": y_test,
            "y_score_test": y_score_test,
        },
        "Policy_B": {
            "X_train": X_train_B,
            "y_train": y_train,
            "y_score_train": y_score_train,
            "X_val": X_val_B,
            "y_val": y_val,
            "y_score_val": y_score_val,
            "X_test": X_test_B,
            "y_test": y_test,
            "y_score_test": y_score_test,
        },
        "meta": {
            "train_n": len(train_df),
            "val_n": len(val_df),
            "test_n": len(test_df),
            "train_transitions_n": int(train_df['has_transition'].sum()),
            "val_transitions_n": int(val_df['has_transition'].sum()),
            "test_transitions_n": int(test_df['has_transition'].sum()),
            "train_users_n": len(train_users),
            "val_users_n": len(val_users),
            "test_users_n": len(test_users),
        }
    }
    return datasets


def train_and_eval_classifier(model, X_train, y_train, X_val, y_val, X_test, y_test, sample_weight=None):
    """
    Fits a classification model and evaluates on validation and test sets.
    """
    start_time = time.time()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    val_probs = model.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = evaluate_classification(y_val, val_preds, y_prob=val_probs)
    
    test_probs = model.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = evaluate_classification(y_test, test_preds, y_prob=test_probs)
    
    return {
        "model": model,
        "fit_time_sec": round(fit_time, 3),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_preds": val_preds,
        "test_preds": test_preds
    }


def compute_calibration_metrics(y_true, y_prob):
    """
    Computes multiclass Brier score, log loss, and classwise Brier score for severe class (Class 4).
    """
    N = len(y_true)
    C = y_prob.shape[1]
    
    # One-hot encode true labels
    y_true_oh = np.zeros((N, C), dtype=np.float32)
    for i, label in enumerate(y_true):
        y_true_oh[i, int(label)] = 1.0
        
    # Multiclass Brier Score: mean squared error over all class probabilities
    brier_score = float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))
    
    # Severe class (Class 4) binary Brier score
    y_severe_true = (y_true == 4).astype(float)
    p_severe = y_prob[:, 4]
    brier_severe = float(brier_score_loss(y_severe_true, p_severe))
    
    # Multi-class Log Loss (cross-entropy)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    y_prob_norm = y_prob_clipped / np.sum(y_prob_clipped, axis=1, keepdims=True)
    loss = float(log_loss(y_true, y_prob_norm, labels=[0, 1, 2, 3, 4]))
    
    return {
        "brier_multiclass": round(brier_score, 4),
        "brier_severe_class4": round(brier_severe, 4),
        "log_loss": round(loss, 4)
    }


def run_probability_calibration(base_model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Compares uncalibrated, Platt sigmoid, and Isotonic calibration.
    Calibrates on Validation probabilities and evaluates on Held-Out Test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    # 1. Uncalibrated raw probabilities
    raw_val_probs = base_model.predict_proba(X_val)
    raw_test_probs = base_model.predict_proba(X_test)
    raw_calib_val = compute_calibration_metrics(y_val, raw_val_probs)
    raw_calib_test = compute_calibration_metrics(y_test, raw_test_probs)
    
    # 2. Platt (Sigmoid) Multiclass Calibration (Logistic Regression on validation probabilities)
    platt_calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    platt_calibrator.fit(raw_val_probs, y_val)
    platt_val_probs = platt_calibrator.predict_proba(raw_val_probs)
    platt_test_probs = platt_calibrator.predict_proba(raw_test_probs)
    platt_calib_val = compute_calibration_metrics(y_val, platt_val_probs)
    platt_calib_test = compute_calibration_metrics(y_test, platt_test_probs)
    
    # 3. Classwise Isotonic Calibration
    iso_val_probs_raw = np.zeros_like(raw_val_probs)
    iso_test_probs_raw = np.zeros_like(raw_test_probs)
    
    for c in range(5):
        iso = IsotonicRegression(out_of_bounds='clip')
        y_val_bin = (y_val == c).astype(float)
        iso.fit(raw_val_probs[:, c], y_val_bin)
        iso_val_probs_raw[:, c] = iso.predict(raw_val_probs[:, c])
        iso_test_probs_raw[:, c] = iso.predict(raw_test_probs[:, c])
        
    # Clip and normalize so probabilities sum to 1.0
    iso_val_probs = np.clip(iso_val_probs_raw, 1e-6, 1.0)
    iso_val_probs = iso_val_probs / np.sum(iso_val_probs, axis=1, keepdims=True)
    
    iso_test_probs = np.clip(iso_test_probs_raw, 1e-6, 1.0)
    iso_test_probs = iso_test_probs / np.sum(iso_test_probs, axis=1, keepdims=True)
    
    iso_calib_val = compute_calibration_metrics(y_val, iso_val_probs)
    iso_calib_test = compute_calibration_metrics(y_test, iso_test_probs)
    
    results = {
        "Uncalibrated": {
            "validation": raw_calib_val,
            "test": raw_calib_test,
            "test_probs": raw_test_probs
        },
        "Platt_Sigmoid": {
            "validation": platt_calib_val,
            "test": platt_calib_test,
            "test_probs": platt_test_probs
        },
        "Isotonic": {
            "validation": iso_calib_val,
            "test": iso_calib_test,
            "test_probs": iso_test_probs
        }
    }
    return results


def run_continuous_score_regression(X_train, y_score_train, X_val, y_score_val, y_cat_val, X_test, y_score_test, y_cat_test):
    """
    Trains XGBoost and HistGradientBoosting Regressors directly on continuous PHQ-9 scores,
    then discretizes predictions into 5 clinical tiers.
    """
    def discretize_phq9(scores):
        cats = np.zeros(len(scores), dtype=int)
        cats[(scores >= 5.0) & (scores < 10.0)] = 1
        cats[(scores >= 10.0) & (scores < 15.0)] = 2
        cats[(scores >= 15.0) & (scores < 20.0)] = 3
        cats[scores >= 20.0] = 4
        return cats
        
    models = {
        "XGBoost_Regressor": xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting_Regressor": HistGradientBoostingRegressor(
            max_iter=150, max_depth=5, learning_rate=0.05, random_state=42
        )
    }
    
    reg_results = {}
    for name, reg in models.items():
        t0 = time.time()
        reg.fit(X_train, y_score_train)
        fit_t = round(time.time() - t0, 3)
        
        pred_val_score = np.clip(reg.predict(X_val), 0.0, 27.0)
        pred_test_score = np.clip(reg.predict(X_test), 0.0, 27.0)
        
        pred_val_cat = discretize_phq9(pred_val_score)
        pred_test_cat = discretize_phq9(pred_test_score)
        
        val_mae = float(mean_absolute_error(y_score_val, pred_val_score))
        test_mae = float(mean_absolute_error(y_score_test, pred_test_score))
        val_rmse = float(root_mean_squared_error(y_score_val, pred_val_score))
        test_rmse = float(root_mean_squared_error(y_score_test, pred_test_score))
        
        val_class_metrics = evaluate_classification(y_cat_val, pred_val_cat)
        test_class_metrics = evaluate_classification(y_cat_test, pred_test_cat)
        
        # Calculate extreme category errors (e.g. true minimal [0] predicted severe [4] or vice versa)
        extreme_val = int(np.sum((y_cat_val == 0) & (pred_val_cat >= 3)) + np.sum((y_cat_val == 4) & (pred_val_cat <= 1)))
        extreme_test = int(np.sum((y_cat_test == 0) & (pred_test_cat >= 3)) + np.sum((y_cat_test == 4) & (pred_test_cat <= 1)))
        
        reg_results[name] = {
            "fit_time_sec": fit_t,
            "continuous_metrics": {
                "val_score_mae": round(val_mae, 4),
                "test_score_mae": round(test_mae, 4),
                "val_score_rmse": round(val_rmse, 4),
                "test_score_rmse": round(test_rmse, 4),
            },
            "discretized_val_metrics": val_class_metrics,
            "discretized_test_metrics": test_class_metrics,
            "extreme_errors_test": extreme_test
        }
        
    return reg_results
