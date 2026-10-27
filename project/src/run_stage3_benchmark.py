import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_dataset, inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from baseline import ClassificationMLP, train_baseline_mlp, prepare_dataset_arrays, compute_class_weights
from evaluation import evaluate_classification, evaluate_regression, save_metrics_to_json
from longitudinal import (
    build_longitudinal_sequences, LongitudinalDataset, collate_seq_batch,
    LongitudinalGRUModel, train_longitudinal_experiment
)
from oversampling import latent_space_smote
from xgboost_model import build_xgboost_classifier, compute_multiclass_sample_weights, train_xgboost_model

# Output directories
METRICS_DIR = "d:/projects/DepDet/project/results/metrics"
PLOTS_DIR = "d:/projects/DepDet/project/results/plots/stage3"
CKPT_DIR = "d:/projects/DepDet/project/checkpoints/stage3"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CLASS_NAMES = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]

# ==============================================================================
# Helper: Feature Group Definitions
# ==============================================================================
def get_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    actigraphy = [c for c in feature_cols if any(k in c.lower() for k in ['step', 'mvpa', 'lpa', 'sedentary', 'active_day', 'light_activity'])]
    sleep = [c for c in feature_cols if any(k in c.lower() for k in ['sleep', 'bed', 'hypersomnia', 'hyposomnia', 'circadian', 'asleep'])]
    rem = [c for c in feature_cols if c not in actigraphy and c not in sleep]
    
    demo_keywords = ['sex', 'race', 'birthyear', 'educ', 'height', 'weight', 'bmi', 'pregnant', 'birth', 'trauma', 'insurance', 'money', 'household']
    demographics = [c for c in rem if any(k in c.lower() for k in demo_keywords)]
    clinical = [c for c in rem if c not in demographics]
    
    return {
        "Actigraphy": actigraphy,
        "Sleep": sleep,
        "Demographics": demographics,
        "Clinical": clinical,
        "Combined": feature_cols
    }

# ==============================================================================
# 1. Benchmark Models Implementation
# ==============================================================================

def train_tuned_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Logistic Regression tuned across C in [0.01, 0.1, 1.0] on Validation Macro-F1."""
    best_c = None
    best_val_f1 = -1.0
    best_clf = None
    best_val_metrics = None
    
    for c_val in [0.01, 0.1, 1.0]:
        clf = LogisticRegression(penalty='l2', C=c_val, max_iter=1000, random_state=42, solver='lbfgs')
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_m = evaluate_classification(y_val, val_preds, y_prob=val_probs)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_c = c_val
            best_clf = clf
            best_val_metrics = val_m
            
    test_probs = best_clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = evaluate_classification(y_test, test_preds, y_prob=test_probs)
    
    return {
        "model_name": "Logistic Regression (L2)",
        "best_config": {"C": best_c, "penalty": "l2"},
        "val": best_val_metrics,
        "test": test_metrics,
        "model": best_clf,
        "val_probs": best_clf.predict_proba(X_val),
        "test_probs": test_probs
    }

def train_tuned_hist_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    """HistGradientBoosting tuned across learning rates and max iterations."""
    configs = [
        {"max_iter": 100, "learning_rate": 0.05, "max_depth": 5},
        {"max_iter": 200, "learning_rate": 0.05, "max_depth": 6},
        {"max_iter": 150, "learning_rate": 0.1, "max_depth": 5}
    ]
    best_cfg = None
    best_val_f1 = -1.0
    best_clf = None
    best_val_metrics = None
    
    for cfg in configs:
        hgb = HistGradientBoostingClassifier(
            max_iter=cfg["max_iter"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10
        )
        hgb.fit(X_train, y_train)
        val_probs = hgb.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_m = evaluate_classification(y_val, val_preds, y_prob=val_probs)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_cfg = cfg
            best_clf = hgb
            best_val_metrics = val_m
            
    test_probs = best_clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = evaluate_classification(y_test, test_preds, y_prob=test_probs)
    
    return {
        "model_name": "HistGradientBoosting",
        "best_config": best_cfg,
        "val": best_val_metrics,
        "test": test_metrics,
        "model": best_clf,
        "val_probs": best_clf.predict_proba(X_val),
        "test_probs": test_probs
    }

def train_tuned_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, use_weights: bool = False):
    """XGBoost multiclass tuned on validation loss and early stopping."""
    sample_weights = compute_multiclass_sample_weights(y_train) if use_weights else None
    
    # Grid of configs
    configs = [
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8}
    ]
    best_val_f1 = -1.0
    best_res = None
    best_model = None
    
    for p in configs:
        model = build_xgboost_classifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            seed=42
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_probs = model.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_m = evaluate_classification(y_val, val_preds, y_prob=val_probs)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_model = model
            test_probs = model.predict_proba(X_test)
            test_preds = np.argmax(test_probs, axis=1)
            test_m = evaluate_classification(y_test, test_preds, y_prob=test_probs)
            best_res = {
                "model_name": "XGBoost (Class-Weighted)" if use_weights else "XGBoost (Unweighted)",
                "best_config": p,
                "val": val_m,
                "test": test_m,
                "model": best_model,
                "val_probs": val_probs,
                "test_probs": test_probs
            }
            
    return best_res

def train_tuned_frank_hall(X_train, y_train, X_val, y_val, X_test, y_test):
    """Frank & Hall Ordinal Classification with monotonic cumulative logits."""
    best_c = None
    best_val_f1 = -1.0
    best_models = None
    best_val_m = None
    best_val_probs = None
    
    for c_val in [0.01, 0.1, 1.0]:
        models = []
        for k in range(4):
            y_tr_bin = (y_train > k).astype(int)
            clf = LogisticRegression(penalty='l2', C=c_val, max_iter=1000, random_state=42 + k)
            clf.fit(X_train, y_tr_bin)
            models.append(clf)
            
        def predict_probs(clf_list, X):
            p_greater = np.zeros((len(X), 4))
            for k in range(4):
                p_greater[:, k] = clf_list[k].predict_proba(X)[:, 1]
            for k in range(1, 4):
                p_greater[:, k] = np.minimum(p_greater[:, k], p_greater[:, k - 1])
            probs = np.zeros((len(X), 5))
            probs[:, 0] = 1.0 - p_greater[:, 0]
            probs[:, 1] = p_greater[:, 0] - p_greater[:, 1]
            probs[:, 2] = p_greater[:, 1] - p_greater[:, 2]
            probs[:, 3] = p_greater[:, 2] - p_greater[:, 3]
            probs[:, 4] = p_greater[:, 3]
            probs = np.clip(probs, 0.0, 1.0)
            probs /= probs.sum(axis=1, keepdims=True)
            return probs

        v_probs = predict_probs(models, X_val)
        v_preds = np.argmax(v_probs, axis=1)
        v_m = evaluate_classification(y_val, v_preds, y_prob=v_probs)
        if v_m["macro_f1"] > best_val_f1:
            best_val_f1 = v_m["macro_f1"]
            best_c = c_val
            best_models = models
            best_val_m = v_m
            best_val_probs = v_probs
            
    def predict_probs_final(clf_list, X):
        p_greater = np.zeros((len(X), 4))
        for k in range(4):
            p_greater[:, k] = clf_list[k].predict_proba(X)[:, 1]
        for k in range(1, 4):
            p_greater[:, k] = np.minimum(p_greater[:, k], p_greater[:, k - 1])
        probs = np.zeros((len(X), 5))
        probs[:, 0] = 1.0 - p_greater[:, 0]
        probs[:, 1] = p_greater[:, 0] - p_greater[:, 1]
        probs[:, 2] = p_greater[:, 1] - p_greater[:, 2]
        probs[:, 3] = p_greater[:, 2] - p_greater[:, 3]
        probs[:, 4] = p_greater[:, 3]
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    t_probs = predict_probs_final(best_models, X_test)
    t_preds = np.argmax(t_probs, axis=1)
    t_m = evaluate_classification(y_test, t_preds, y_prob=t_probs)
    
    return {
        "model_name": "Frank & Hall Ordinal Classifier",
        "best_config": {"C": best_c},
        "val": best_val_m,
        "test": t_m,
        "model": best_models,
        "val_probs": best_val_probs,
        "test_probs": t_probs
    }

def train_latent_smote_benchmark(train_df, val_df, test_df, f_cols, X_train, y_train, X_val, y_val, X_test, y_test, unweighted_encoder):
    """Corrected Latent-SMOTE using embeddings extracted from trained unweighted MLP."""
    unweighted_encoder.eval()
    with torch.no_grad():
        _, _, lat_tr = unweighted_encoder(torch.tensor(X_train, dtype=torch.float32))
        _, _, lat_va = unweighted_encoder(torch.tensor(X_val, dtype=torch.float32))
        _, _, lat_te = unweighted_encoder(torch.tensor(X_test, dtype=torch.float32))
        
    lat_tr_np = lat_tr.numpy()
    lat_va_np = lat_va.numpy()
    lat_te_np = lat_te.numpy()
    
    bal_lat_tr, bal_y_tr = latent_space_smote(lat_tr_np, y_train)
    
    clf_head = nn.Linear(bal_lat_tr.shape[1], 5)
    opt = torch.optim.Adam(clf_head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    X_smote_t = torch.tensor(bal_lat_tr, dtype=torch.float32)
    y_smote_t = torch.tensor(bal_y_tr, dtype=torch.long)
    
    best_smote_f1 = -1.0
    best_smote_weights = None
    
    for epoch in range(1, 40):
        clf_head.train()
        opt.zero_grad()
        out = clf_head(X_smote_t)
        loss = crit(out, y_smote_t)
        loss.backward()
        opt.step()
        
        clf_head.eval()
        with torch.no_grad():
            v_logits = clf_head(torch.tensor(lat_va_np, dtype=torch.float32))
            v_probs = torch.softmax(v_logits, dim=1).numpy()
            v_pred = np.argmax(v_probs, axis=1)
            v_cls = evaluate_classification(y_val, v_pred, y_prob=v_probs)
            if v_cls["macro_f1"] > best_smote_f1:
                best_smote_f1 = v_cls["macro_f1"]
                best_smote_weights = copy.deepcopy(clf_head.state_dict())
                
    if best_smote_weights is not None:
        clf_head.load_state_dict(best_smote_weights)
        
    clf_head.eval()
    with torch.no_grad():
        v_final_logits = clf_head(torch.tensor(lat_va_np, dtype=torch.float32))
        v_final_probs = torch.softmax(v_final_logits, dim=1).numpy()
        val_preds_smote = np.argmax(v_final_probs, axis=1)
        
        t_final_logits = clf_head(torch.tensor(lat_te_np, dtype=torch.float32))
        t_final_probs = torch.softmax(t_final_logits, dim=1).numpy()
        test_preds_smote = np.argmax(t_final_probs, axis=1)
        
    val_m = evaluate_classification(y_val, val_preds_smote, y_prob=v_final_probs)
    test_m = evaluate_classification(y_test, test_preds_smote, y_prob=t_final_probs)
    
    return {
        "model_name": "Corrected Latent-SMOTE MLP",
        "best_config": {"encoder": "Unweighted Classification MLP", "smote_space": "latent_64d"},
        "val": val_m,
        "test": test_m,
        "model": (unweighted_encoder, clf_head),
        "val_probs": v_final_probs,
        "test_probs": t_final_probs
    }

# ==============================================================================
# 2. Main Stage 3 Execution Function
# ==============================================================================

def main():
    print("=" * 80)
    print("[STAGE 3] LAUNCHING STAGE 3: CONTROLLED BENCHMARK EXPERIMENTS & ANALYSIS")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Load Dataset
    parquet_path = "d:/projects/DepDet/anon_processed_df_parquet"
    df = load_raw_dataset(parquet_path)
    feature_cols, num_cols, cat_cols, target_cols, inspection_summary = inspect_modalities_and_features(df)
    
    print(f"Total Rows: {len(df)}, Features: {len(feature_cols)}, Participants: {df['user_id'].nunique()}")
    
    # 2. Perform Subject-Level Train/Val/Test Split (70 / 15 / 15)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df, seed=42)
    print(f"Train: {len(train_df)} rows ({split_info['n_train_participants']} subjects)")
    print(f"Val:   {len(val_df)} rows ({split_info['n_val_participants']} subjects)")
    print(f"Test:  {len(test_df)} rows ({split_info['n_test_participants']} subjects)")
    
    # Preprocessing
    prep = DataPreprocessor(feature_cols=feature_cols)
    X_train, y_train, y_train_score, _, _ = prepare_dataset_arrays(train_df, prep, fit_preprocessor=True)
    X_val, y_val, y_val_score, _, _ = prepare_dataset_arrays(val_df, prep, fit_preprocessor=False)
    X_test, y_test, y_test_score, _, _ = prepare_dataset_arrays(test_df, prep, fit_preprocessor=False)
    
    benchmark_results = []
    per_class_all = {}
    confusion_matrices_all = {}
    
    # --------------------------------------------------------------------------
    # Model 1: Logistic Regression
    # --------------------------------------------------------------------------
    print("\n[1/11] Running Logistic Regression (L2)...")
    res_lr = train_tuned_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    benchmark_results.append({
        "Model": "Logistic Regression (L2)",
        "Category": "Linear Baseline",
        "Val Macro-F1": res_lr["val"]["macro_f1"],
        "Val Bal-Acc": res_lr["val"]["balanced_accuracy"],
        "Val Accuracy": res_lr["val"]["accuracy"],
        "Val G-Mean": res_lr["val"]["multiclass_g_mean"],
        "Val QWK": res_lr["val"]["quadratic_weighted_kappa"],
        "Val Ordinal MAE": res_lr["val"]["ordinal_class_mae"],
        "Val OVR ROC-AUC": res_lr["val"]["ovr_roc_auc_macro"],
        "Test Macro-F1": res_lr["test"]["macro_f1"],
        "Test Bal-Acc": res_lr["test"]["balanced_accuracy"],
        "Test Accuracy": res_lr["test"]["accuracy"],
        "Test G-Mean": res_lr["test"]["multiclass_g_mean"],
        "Test QWK": res_lr["test"]["quadratic_weighted_kappa"],
        "Test Ordinal MAE": res_lr["test"]["ordinal_class_mae"],
        "Test OVR ROC-AUC": res_lr["test"]["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["Logistic Regression"] = res_lr["test"]["per_class_f1"]
    confusion_matrices_all["Logistic Regression"] = res_lr["test"]["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 2: Unweighted Classification MLP
    # --------------------------------------------------------------------------
    print("\n[2/11] Running Unweighted Classification MLP...")
    model_e2, _, res_mlp, _, _ = train_baseline_mlp(
        train_df, val_df, test_df, feature_cols,
        use_class_weights=False, aux_reg_weight=0.0,
        epochs=30, patience=7, seed=42, checkpoint_name="stage3_unweighted_mlp.pt"
    )
    val_m2 = res_mlp["val_classification"]
    test_m2 = res_mlp["test_classification"]
    benchmark_results.append({
        "Model": "Unweighted Classification MLP",
        "Category": "Neural Baseline",
        "Val Macro-F1": val_m2["macro_f1"],
        "Val Bal-Acc": val_m2["balanced_accuracy"],
        "Val Accuracy": val_m2["accuracy"],
        "Val G-Mean": val_m2["multiclass_g_mean"],
        "Val QWK": val_m2["quadratic_weighted_kappa"],
        "Val Ordinal MAE": val_m2["ordinal_class_mae"],
        "Val OVR ROC-AUC": val_m2["ovr_roc_auc_macro"],
        "Test Macro-F1": test_m2["macro_f1"],
        "Test Bal-Acc": test_m2["balanced_accuracy"],
        "Test Accuracy": test_m2["accuracy"],
        "Test G-Mean": test_m2["multiclass_g_mean"],
        "Test QWK": test_m2["quadratic_weighted_kappa"],
        "Test Ordinal MAE": test_m2["ordinal_class_mae"],
        "Test OVR ROC-AUC": test_m2["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["Unweighted MLP"] = test_m2["per_class_f1"]
    confusion_matrices_all["Unweighted MLP"] = test_m2["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 3: Class-Weighted MLP
    # --------------------------------------------------------------------------
    print("\n[3/11] Running Class-Weighted MLP...")
    model_e3, _, res_w_mlp, _, _ = train_baseline_mlp(
        train_df, val_df, test_df, feature_cols,
        use_class_weights=True, aux_reg_weight=0.0,
        epochs=30, patience=7, seed=42, checkpoint_name="stage3_class_weighted_mlp.pt"
    )
    val_m3 = res_w_mlp["val_classification"]
    test_m3 = res_w_mlp["test_classification"]
    benchmark_results.append({
        "Model": "Class-Weighted MLP",
        "Category": "Imbalance Handling",
        "Val Macro-F1": val_m3["macro_f1"],
        "Val Bal-Acc": val_m3["balanced_accuracy"],
        "Val Accuracy": val_m3["accuracy"],
        "Val G-Mean": val_m3["multiclass_g_mean"],
        "Val QWK": val_m3["quadratic_weighted_kappa"],
        "Val Ordinal MAE": val_m3["ordinal_class_mae"],
        "Val OVR ROC-AUC": val_m3["ovr_roc_auc_macro"],
        "Test Macro-F1": test_m3["macro_f1"],
        "Test Bal-Acc": test_m3["balanced_accuracy"],
        "Test Accuracy": test_m3["accuracy"],
        "Test G-Mean": test_m3["multiclass_g_mean"],
        "Test QWK": test_m3["quadratic_weighted_kappa"],
        "Test Ordinal MAE": test_m3["ordinal_class_mae"],
        "Test OVR ROC-AUC": test_m3["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["Class-Weighted MLP"] = test_m3["per_class_f1"]
    confusion_matrices_all["Class-Weighted MLP"] = test_m3["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 4: Corrected Latent-SMOTE MLP
    # --------------------------------------------------------------------------
    print("\n[4/11] Running Corrected Latent-SMOTE MLP...")
    res_smote = train_latent_smote_benchmark(
        train_df, val_df, test_df, feature_cols,
        X_train, y_train, X_val, y_val, X_test, y_test,
        unweighted_encoder=model_e2
    )
    benchmark_results.append({
        "Model": "Corrected Latent-SMOTE MLP",
        "Category": "Imbalance Handling",
        "Val Macro-F1": res_smote["val"]["macro_f1"],
        "Val Bal-Acc": res_smote["val"]["balanced_accuracy"],
        "Val Accuracy": res_smote["val"]["accuracy"],
        "Val G-Mean": res_smote["val"]["multiclass_g_mean"],
        "Val QWK": res_smote["val"]["quadratic_weighted_kappa"],
        "Val Ordinal MAE": res_smote["val"]["ordinal_class_mae"],
        "Val OVR ROC-AUC": res_smote["val"]["ovr_roc_auc_macro"],
        "Test Macro-F1": res_smote["test"]["macro_f1"],
        "Test Bal-Acc": res_smote["test"]["balanced_accuracy"],
        "Test Accuracy": res_smote["test"]["accuracy"],
        "Test G-Mean": res_smote["test"]["multiclass_g_mean"],
        "Test QWK": res_smote["test"]["quadratic_weighted_kappa"],
        "Test Ordinal MAE": res_smote["test"]["ordinal_class_mae"],
        "Test OVR ROC-AUC": res_smote["test"]["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["Latent SMOTE"] = res_smote["test"]["per_class_f1"]
    confusion_matrices_all["Latent SMOTE"] = res_smote["test"]["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 5: HistGradientBoosting
    # --------------------------------------------------------------------------
    print("\n[5/11] Running HistGradientBoosting...")
    res_hgb = train_tuned_hist_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test)
    benchmark_results.append({
        "Model": "HistGradientBoosting",
        "Category": "Tabular Tree Baseline",
        "Val Macro-F1": res_hgb["val"]["macro_f1"],
        "Val Bal-Acc": res_hgb["val"]["balanced_accuracy"],
        "Val Accuracy": res_hgb["val"]["accuracy"],
        "Val G-Mean": res_hgb["val"]["multiclass_g_mean"],
        "Val QWK": res_hgb["val"]["quadratic_weighted_kappa"],
        "Val Ordinal MAE": res_hgb["val"]["ordinal_class_mae"],
        "Val OVR ROC-AUC": res_hgb["val"]["ovr_roc_auc_macro"],
        "Test Macro-F1": res_hgb["test"]["macro_f1"],
        "Test Bal-Acc": res_hgb["test"]["balanced_accuracy"],
        "Test Accuracy": res_hgb["test"]["accuracy"],
        "Test G-Mean": res_hgb["test"]["multiclass_g_mean"],
        "Test QWK": res_hgb["test"]["quadratic_weighted_kappa"],
        "Test Ordinal MAE": res_hgb["test"]["ordinal_class_mae"],
        "Test OVR ROC-AUC": res_hgb["test"]["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["HistGradientBoosting"] = res_hgb["test"]["per_class_f1"]
    confusion_matrices_all["HistGradientBoosting"] = res_hgb["test"]["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 6: XGBoost Multiclass Classifier (Unweighted & Weighted)
    # --------------------------------------------------------------------------
    print("\n[6/11] Running XGBoost Multiclass (Unweighted & Weighted)...")
    res_xgb_unw = train_tuned_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, use_weights=False)
    benchmark_results.append({
        "Model": "XGBoost Multiclass",
        "Category": "Tabular Tree Baseline",
        "Val Macro-F1": res_xgb_unw["val"]["macro_f1"],
        "Val Bal-Acc": res_xgb_unw["val"]["balanced_accuracy"],
        "Val Accuracy": res_xgb_unw["val"]["accuracy"],
        "Val G-Mean": res_xgb_unw["val"]["multiclass_g_mean"],
        "Val QWK": res_xgb_unw["val"]["quadratic_weighted_kappa"],
        "Val Ordinal MAE": res_xgb_unw["val"]["ordinal_class_mae"],
        "Val OVR ROC-AUC": res_xgb_unw["val"]["ovr_roc_auc_macro"],
        "Test Macro-F1": res_xgb_unw["test"]["macro_f1"],
        "Test Bal-Acc": res_xgb_unw["test"]["balanced_accuracy"],
        "Test Accuracy": res_xgb_unw["test"]["accuracy"],
        "Test G-Mean": res_xgb_unw["test"]["multiclass_g_mean"],
        "Test QWK": res_xgb_unw["test"]["quadratic_weighted_kappa"],
        "Test Ordinal MAE": res_xgb_unw["test"]["ordinal_class_mae"],
        "Test OVR ROC-AUC": res_xgb_unw["test"]["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["XGBoost"] = res_xgb_unw["test"]["per_class_f1"]
    confusion_matrices_all["XGBoost"] = res_xgb_unw["test"]["confusion_matrix"]
    
    res_xgb_w = train_tuned_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, use_weights=True)
    benchmark_results.append({
        "Model": "XGBoost (Class-Weighted)",
        "Category": "Imbalance Handling",
        "Val Macro-F1": res_xgb_w["val"]["macro_f1"],
        "Val Bal-Acc": res_xgb_w["val"]["balanced_accuracy"],
        "Val Accuracy": res_xgb_w["val"]["accuracy"],
        "Val G-Mean": res_xgb_w["val"]["multiclass_g_mean"],
        "Val QWK": res_xgb_w["val"]["quadratic_weighted_kappa"],
        "Val Ordinal MAE": res_xgb_w["val"]["ordinal_class_mae"],
        "Val OVR ROC-AUC": res_xgb_w["val"]["ovr_roc_auc_macro"],
        "Test Macro-F1": res_xgb_w["test"]["macro_f1"],
        "Test Bal-Acc": res_xgb_w["test"]["balanced_accuracy"],
        "Test Accuracy": res_xgb_w["test"]["accuracy"],
        "Test G-Mean": res_xgb_w["test"]["multiclass_g_mean"],
        "Test QWK": res_xgb_w["test"]["quadratic_weighted_kappa"],
        "Test Ordinal MAE": res_xgb_w["test"]["ordinal_class_mae"],
        "Test OVR ROC-AUC": res_xgb_w["test"]["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["XGBoost Weighted"] = res_xgb_w["test"]["per_class_f1"]
    confusion_matrices_all["XGBoost Weighted"] = res_xgb_w["test"]["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 7: Frank–Hall Ordinal Classifier
    # --------------------------------------------------------------------------
    print("\n[7/11] Running Frank–Hall Ordinal Classifier...")
    res_fh = train_tuned_frank_hall(X_train, y_train, X_val, y_val, X_test, y_test)
    benchmark_results.append({
        "Model": "Frank–Hall Ordinal Classifier",
        "Category": "Ordinal Decomposition",
        "Val Macro-F1": res_fh["val"]["macro_f1"],
        "Val Bal-Acc": res_fh["val"]["balanced_accuracy"],
        "Val Accuracy": res_fh["val"]["accuracy"],
        "Val G-Mean": res_fh["val"]["multiclass_g_mean"],
        "Val QWK": res_fh["val"]["quadratic_weighted_kappa"],
        "Val Ordinal MAE": res_fh["val"]["ordinal_class_mae"],
        "Val OVR ROC-AUC": res_fh["val"]["ovr_roc_auc_macro"],
        "Test Macro-F1": res_fh["test"]["macro_f1"],
        "Test Bal-Acc": res_fh["test"]["balanced_accuracy"],
        "Test Accuracy": res_fh["test"]["accuracy"],
        "Test G-Mean": res_fh["test"]["multiclass_g_mean"],
        "Test QWK": res_fh["test"]["quadratic_weighted_kappa"],
        "Test Ordinal MAE": res_fh["test"]["ordinal_class_mae"],
        "Test OVR ROC-AUC": res_fh["test"]["ovr_roc_auc_macro"],
        "Test Reg MAE": None
    })
    per_class_all["Frank-Hall Ordinal"] = res_fh["test"]["per_class_f1"]
    confusion_matrices_all["Frank-Hall Ordinal"] = res_fh["test"]["confusion_matrix"]
    
    # --------------------------------------------------------------------------
    # Model 8-10: Longitudinal GRU Models (Current, History, Fusion)
    # --------------------------------------------------------------------------
    print("\n[8-10/11] Running Longitudinal GRU Models (Current, History, Fusion)...")
    res_curr = train_longitudinal_experiment(train_df, val_df, test_df, feature_cols, mode="current_only", epochs=25, seed=42)
    res_hist = train_longitudinal_experiment(train_df, val_df, test_df, feature_cols, mode="history_only", epochs=25, seed=42)
    res_fuse = train_longitudinal_experiment(train_df, val_df, test_df, feature_cols, mode="current_plus_history", epochs=25, seed=42)
    
    for label, r in [("Longitudinal GRU (Current Only)", res_curr),
                     ("Longitudinal GRU (History Only)", res_hist),
                     ("Longitudinal GRU (Current+History Fusion)", res_fuse)]:
        v = r["val_classification"]
        t = r["test_classification"]
        benchmark_results.append({
            "Model": label,
            "Category": "Longitudinal Recurrent",
            "Val Macro-F1": v["macro_f1"],
            "Val Bal-Acc": v["balanced_accuracy"],
            "Val Accuracy": v["accuracy"],
            "Val G-Mean": v["multiclass_g_mean"],
            "Val QWK": v["quadratic_weighted_kappa"],
            "Val Ordinal MAE": v["ordinal_class_mae"],
            "Val OVR ROC-AUC": v.get("ovr_roc_auc_macro"),
            "Test Macro-F1": t["macro_f1"],
            "Test Bal-Acc": t["balanced_accuracy"],
            "Test Accuracy": t["accuracy"],
            "Test G-Mean": t["multiclass_g_mean"],
            "Test QWK": t["quadratic_weighted_kappa"],
            "Test Ordinal MAE": t["ordinal_class_mae"],
            "Test OVR ROC-AUC": t.get("ovr_roc_auc_macro"),
            "Test Reg MAE": r["test_regression"]["mae"] if r.get("test_regression") else None
        })
        per_class_all[label] = t["per_class_f1"]
        confusion_matrices_all[label] = t["confusion_matrix"]
        
    # --------------------------------------------------------------------------
    # Model 11: Corrected Auxiliary-Regression MLP Grid (Tuning lambda in [0.1, 0.2, 0.5])
    # --------------------------------------------------------------------------
    print("\n[11/11] Running Corrected Auxiliary-Regression MLP Grid...")
    best_aux_w = None
    best_aux_val_f1 = -1.0
    best_aux_res = None
    
    for w in [0.1, 0.2, 0.5]:
        _, _, metrics_w, _, _ = train_baseline_mlp(
            train_df, val_df, test_df, feature_cols,
            use_class_weights=False, aux_reg_weight=w,
            epochs=25, patience=6, seed=42, checkpoint_name=f"stage3_aux_reg_w{w}.pt"
        )
        if metrics_w["val_classification"]["macro_f1"] > best_aux_val_f1:
            best_aux_val_f1 = metrics_w["val_classification"]["macro_f1"]
            best_aux_w = w
            best_aux_res = metrics_w
            
    v_aux = best_aux_res["val_classification"]
    t_aux = best_aux_res["test_classification"]
    benchmark_results.append({
        "Model": f"Multi-Task Auxiliary-Reg MLP (w={best_aux_w})",
        "Category": "Multi-Task Regularization",
        "Val Macro-F1": v_aux["macro_f1"],
        "Val Bal-Acc": v_aux["balanced_accuracy"],
        "Val Accuracy": v_aux["accuracy"],
        "Val G-Mean": v_aux["multiclass_g_mean"],
        "Val QWK": v_aux["quadratic_weighted_kappa"],
        "Val Ordinal MAE": v_aux["ordinal_class_mae"],
        "Val OVR ROC-AUC": v_aux.get("ovr_roc_auc_macro"),
        "Test Macro-F1": t_aux["macro_f1"],
        "Test Bal-Acc": t_aux["balanced_accuracy"],
        "Test Accuracy": t_aux["accuracy"],
        "Test G-Mean": t_aux["multiclass_g_mean"],
        "Test QWK": t_aux["quadratic_weighted_kappa"],
        "Test Ordinal MAE": t_aux["ordinal_class_mae"],
        "Test OVR ROC-AUC": t_aux.get("ovr_roc_auc_macro"),
        "Test Reg MAE": best_aux_res["test_regression"]["mae"] if best_aux_res.get("test_regression") else None
    })
    per_class_all[f"Aux-Reg MLP (w={best_aux_w})"] = t_aux["per_class_f1"]
    confusion_matrices_all[f"Aux-Reg MLP (w={best_aux_w})"] = t_aux["confusion_matrix"]
    
    # Save Unified Benchmark Table
    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv(os.path.join(METRICS_DIR, "stage3_benchmark_results.csv"), index=False)
    save_metrics_to_json(benchmark_results, os.path.join(METRICS_DIR, "stage3_benchmark_results.json"))
    save_metrics_to_json(per_class_all, os.path.join(METRICS_DIR, "stage3_per_class_metrics.json"))
    save_metrics_to_json(confusion_matrices_all, os.path.join(METRICS_DIR, "stage3_confusion_matrices.json"))
    
    print("\n" + "=" * 80)
    print("STAGE 3 BENCHMARK RESULTS TABLE:")
    print("=" * 80)
    print(benchmark_df[["Model", "Val Macro-F1", "Val Bal-Acc", "Test Macro-F1", "Test Bal-Acc", "Test Accuracy", "Test G-Mean", "Test QWK"]].to_string(index=False))
    
    # ==============================================================================
    # 4. Fair Longitudinal Comparison (Matched vs Full Cohort)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("[STAGE 3] RUNNING FAIR LONGITUDINAL COMPARISON (MATCHED VS FULL COHORT)")
    print("=" * 80)
    
    test_samples = build_longitudinal_sequences(test_df, prep)
    matched_test_samples = [s for s in test_samples if s["history_len"] > 0]
    matched_test_users = set(s["user_id"] for s in matched_test_samples)
    full_test_users = set(s["user_id"] for s in test_samples)
    
    print(f"Full Test Cohort:    N = {len(test_samples)} assessments across {len(full_test_users)} participants")
    print(f"Matched Test Cohort: N = {len(matched_test_samples)} assessments across {len(matched_test_users)} participants (history_len >= 1)")
    
    # Create DataLoader for matched cohort
    matched_ds = LongitudinalDataset(matched_test_samples, require_history=True)
    matched_loader = DataLoader(matched_ds, batch_size=128, shuffle=False, collate_fn=collate_seq_batch)
    
    matched_comparison = []
    
    # Evaluate Current Only on Matched
    ckpt_curr = torch.load("d:/projects/DepDet/project/checkpoints/longitudinal_current_only_corrected.pt", weights_only=True)
    model_curr = LongitudinalGRUModel(input_dim=len(feature_cols), hidden_dim=128, num_classes=5, mode="current_only")
    model_curr.load_state_dict(ckpt_curr)
    model_curr.eval()
    
    c_preds, c_trues = [], []
    with torch.no_grad():
        for b in matched_loader:
            logits, _, _ = model_curr(b["curr_X"], b["history_X"], b["lengths"], b["has_history"])
            c_preds.extend(torch.argmax(logits, dim=1).numpy())
            c_trues.extend(b["y_cat"].numpy())
    m_curr_matched = evaluate_classification(np.array(c_trues), np.array(c_preds))
    matched_comparison.append({
        "Model": "Longitudinal Current Only",
        "Evaluation Population": "Matched Cohort (history_len >= 1)",
        "N_Samples": len(matched_test_samples),
        "N_Participants": len(matched_test_users),
        "Macro-F1": m_curr_matched["macro_f1"],
        "Balanced Accuracy": m_curr_matched["balanced_accuracy"],
        "Accuracy": m_curr_matched["accuracy"],
        "G-Mean": m_curr_matched["multiclass_g_mean"],
        "QWK": m_curr_matched["quadratic_weighted_kappa"],
        "Ordinal MAE": m_curr_matched["ordinal_class_mae"]
    })
    
    # Evaluate History Only on Matched
    ckpt_hist = torch.load("d:/projects/DepDet/project/checkpoints/longitudinal_history_only_corrected.pt", weights_only=True)
    model_hist = LongitudinalGRUModel(input_dim=len(feature_cols), hidden_dim=128, num_classes=5, mode="history_only")
    model_hist.load_state_dict(ckpt_hist)
    model_hist.eval()
    
    h_preds, h_trues = [], []
    with torch.no_grad():
        for b in matched_loader:
            logits, _, _ = model_hist(b["curr_X"], b["history_X"], b["lengths"], b["has_history"])
            h_preds.extend(torch.argmax(logits, dim=1).numpy())
            h_trues.extend(b["y_cat"].numpy())
    m_hist_matched = evaluate_classification(np.array(h_trues), np.array(h_preds))
    matched_comparison.append({
        "Model": "Longitudinal History Only",
        "Evaluation Population": "Matched Cohort (history_len >= 1)",
        "N_Samples": len(matched_test_samples),
        "N_Participants": len(matched_test_users),
        "Macro-F1": m_hist_matched["macro_f1"],
        "Balanced Accuracy": m_hist_matched["balanced_accuracy"],
        "Accuracy": m_hist_matched["accuracy"],
        "G-Mean": m_hist_matched["multiclass_g_mean"],
        "QWK": m_hist_matched["quadratic_weighted_kappa"],
        "Ordinal MAE": m_hist_matched["ordinal_class_mae"]
    })
    
    # Evaluate Current + History Fusion on Matched
    ckpt_fuse = torch.load("d:/projects/DepDet/project/checkpoints/longitudinal_current_plus_history_corrected.pt", weights_only=True)
    model_fuse = LongitudinalGRUModel(input_dim=len(feature_cols), hidden_dim=128, num_classes=5, mode="current_plus_history")
    model_fuse.load_state_dict(ckpt_fuse)
    model_fuse.eval()
    
    f_preds, f_trues = [], []
    with torch.no_grad():
        for b in matched_loader:
            logits, _, _ = model_fuse(b["curr_X"], b["history_X"], b["lengths"], b["has_history"])
            f_preds.extend(torch.argmax(logits, dim=1).numpy())
            f_trues.extend(b["y_cat"].numpy())
    m_fuse_matched = evaluate_classification(np.array(f_trues), np.array(f_preds))
    matched_comparison.append({
        "Model": "Longitudinal Current+History Fusion",
        "Evaluation Population": "Matched Cohort (history_len >= 1)",
        "N_Samples": len(matched_test_samples),
        "N_Participants": len(matched_test_users),
        "Macro-F1": m_fuse_matched["macro_f1"],
        "Balanced Accuracy": m_fuse_matched["balanced_accuracy"],
        "Accuracy": m_fuse_matched["accuracy"],
        "G-Mean": m_fuse_matched["multiclass_g_mean"],
        "QWK": m_fuse_matched["quadratic_weighted_kappa"],
        "Ordinal MAE": m_fuse_matched["ordinal_class_mae"]
    })
    
    # Full Cohort Rows for Current and Fusion
    matched_comparison.append({
        "Model": "Longitudinal Current Only",
        "Evaluation Population": "Full Cohort (All Periods)",
        "N_Samples": len(test_samples),
        "N_Participants": len(full_test_users),
        "Macro-F1": res_curr["test_classification"]["macro_f1"],
        "Balanced Accuracy": res_curr["test_classification"]["balanced_accuracy"],
        "Accuracy": res_curr["test_classification"]["accuracy"],
        "G-Mean": res_curr["test_classification"]["multiclass_g_mean"],
        "QWK": res_curr["test_classification"]["quadratic_weighted_kappa"],
        "Ordinal MAE": res_curr["test_classification"]["ordinal_class_mae"]
    })
    matched_comparison.append({
        "Model": "Longitudinal Current+History Fusion",
        "Evaluation Population": "Full Cohort (All Periods)",
        "N_Samples": len(test_samples),
        "N_Participants": len(full_test_users),
        "Macro-F1": res_fuse["test_classification"]["macro_f1"],
        "Balanced Accuracy": res_fuse["test_classification"]["balanced_accuracy"],
        "Accuracy": res_fuse["test_classification"]["accuracy"],
        "G-Mean": res_fuse["test_classification"]["multiclass_g_mean"],
        "QWK": res_fuse["test_classification"]["quadratic_weighted_kappa"],
        "Ordinal MAE": res_fuse["test_classification"]["ordinal_class_mae"]
    })
    
    matched_df = pd.DataFrame(matched_comparison)
    matched_df.to_csv(os.path.join(METRICS_DIR, "stage3_matched_longitudinal_results.csv"), index=False)
    save_metrics_to_json(matched_comparison, os.path.join(METRICS_DIR, "stage3_matched_longitudinal_results.json"))
    
    print("\nMATCHED VS FULL LONGITUDINAL COMPARISON TABLE:")
    print(matched_df.to_string(index=False))
    
    # ==============================================================================
    # 5. Feature-Group Ablation
    # ==============================================================================
    print("\n" + "=" * 80)
    print("[STAGE 3] RUNNING FEATURE-GROUP ABLATIONS (ACTIGRAPHY, SLEEP, DEMO, CLINICAL)")
    print("=" * 80)
    
    feature_groups = get_feature_groups(feature_cols)
    ablation_results = []
    
    for group_name, cols in feature_groups.items():
        print(f"\nEvaluating Feature Group: {group_name} ({len(cols)} columns)...")
        prep_grp = DataPreprocessor(feature_cols=cols)
        X_tr_grp, y_tr_grp, _, _, _ = prepare_dataset_arrays(train_df, prep_grp, fit_preprocessor=True)
        X_va_grp, y_va_grp, _, _, _ = prepare_dataset_arrays(val_df, prep_grp, fit_preprocessor=False)
        X_te_grp, y_te_grp, _, _, _ = prepare_dataset_arrays(test_df, prep_grp, fit_preprocessor=False)
        
        # Train tuned Logistic Regression on this feature subset
        res_grp = train_tuned_logistic_regression(X_tr_grp, y_tr_grp, X_va_grp, y_va_grp, X_te_grp, y_te_grp)
        v_g = res_grp["val"]
        t_g = res_grp["test"]
        
        ablation_results.append({
            "Feature Group": group_name,
            "N_Features": len(cols),
            "Val Macro-F1": v_g["macro_f1"],
            "Val Bal-Acc": v_g["balanced_accuracy"],
            "Val Accuracy": v_g["accuracy"],
            "Test Macro-F1": t_g["macro_f1"],
            "Test Bal-Acc": t_g["balanced_accuracy"],
            "Test Accuracy": t_g["accuracy"],
            "Test G-Mean": t_g["multiclass_g_mean"],
            "Test QWK": t_g["quadratic_weighted_kappa"],
            "Test Ordinal MAE": t_g["ordinal_class_mae"]
        })
        
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(METRICS_DIR, "stage3_ablation_results.csv"), index=False)
    save_metrics_to_json(ablation_results, os.path.join(METRICS_DIR, "stage3_ablation_results.json"))
    
    print("\nFEATURE-GROUP ABLATION SUMMARY TABLE:")
    print(ablation_df.to_string(index=False))
    
    # ==============================================================================
    # 6. Multi-Seed Stability (Seeds: 42, 123, 2026 on Fixed Split)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("[STAGE 3] RUNNING MULTI-SEED STABILITY EVALUATION (SEEDS 42, 123, 2026)")
    print("=" * 80)
    
    seeds = [42, 123, 2026]
    models_to_test = ["Logistic Regression", "Unweighted MLP", "Class-Weighted MLP", "XGBoost"]
    multiseed_records = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        
        # Logistic Regression
        clf_lr = LogisticRegression(penalty='l2', C=0.1, max_iter=1000, random_state=seed, solver='lbfgs')
        clf_lr.fit(X_train, y_train)
        preds_lr = clf_lr.predict(X_test)
        m_lr = evaluate_classification(y_test, preds_lr)
        multiseed_records.append({"Model": "Logistic Regression", "Seed": seed, "Macro-F1": m_lr["macro_f1"], "Bal-Acc": m_lr["balanced_accuracy"], "Accuracy": m_lr["accuracy"]})
        
        # Unweighted MLP
        _, _, r_unw, _, _ = train_baseline_mlp(train_df, val_df, test_df, feature_cols, use_class_weights=False, aux_reg_weight=0.0, epochs=25, patience=5, seed=seed)
        m_unw = r_unw["test_classification"]
        multiseed_records.append({"Model": "Unweighted MLP", "Seed": seed, "Macro-F1": m_unw["macro_f1"], "Bal-Acc": m_unw["balanced_accuracy"], "Accuracy": m_unw["accuracy"]})
        
        # Class-Weighted MLP
        _, _, r_w, _, _ = train_baseline_mlp(train_df, val_df, test_df, feature_cols, use_class_weights=True, aux_reg_weight=0.0, epochs=25, patience=5, seed=seed)
        m_w = r_w["test_classification"]
        multiseed_records.append({"Model": "Class-Weighted MLP", "Seed": seed, "Macro-F1": m_w["macro_f1"], "Bal-Acc": m_w["balanced_accuracy"], "Accuracy": m_w["accuracy"]})
        
        # XGBoost
        w_train = compute_multiclass_sample_weights(y_train)
        xgb_m = build_xgboost_classifier(n_estimators=150, max_depth=5, learning_rate=0.05, seed=seed)
        xgb_m.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        p_xgb = xgb_m.predict(X_test)
        m_xgb = evaluate_classification(y_test, p_xgb)
        multiseed_records.append({"Model": "XGBoost", "Seed": seed, "Macro-F1": m_xgb["macro_f1"], "Bal-Acc": m_xgb["balanced_accuracy"], "Accuracy": m_xgb["accuracy"]})
        
    ms_df = pd.DataFrame(multiseed_records)
    summary_ms = ms_df.groupby("Model").agg(
        Mean_Macro_F1=("Macro-F1", "mean"),
        Std_Macro_F1=("Macro-F1", "std"),
        Mean_Bal_Acc=("Bal-Acc", "mean"),
        Std_Bal_Acc=("Bal-Acc", "std"),
        Mean_Accuracy=("Accuracy", "mean"),
        Std_Accuracy=("Accuracy", "std")
    ).reset_index()
    
    ms_df.to_csv(os.path.join(METRICS_DIR, "stage3_multiseed_raw_runs.csv"), index=False)
    summary_ms.to_csv(os.path.join(METRICS_DIR, "stage3_multiseed_summary.csv"), index=False)
    save_metrics_to_json(summary_ms.to_dict(orient="records"), os.path.join(METRICS_DIR, "stage3_multiseed_results.json"))
    
    print("\nMULTI-SEED STABILITY SUMMARY TABLE (3 SEEDS):")
    print(summary_ms.to_string(index=False))
    
    # ==============================================================================
    # 7. Explainability & Visualizations
    # ==============================================================================
    print("\n" + "=" * 80)
    print("[STAGE 3] GENERATING PUBLICATION-QUALITY FIGURES AND EXPLAINABILITY ARTIFACTS")
    print("=" * 80)
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: Confusion Matrices for Top Models (Class-Weighted MLP vs Unweighted MLP vs Frank-Hall)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (m_name, cm_key) in zip(axes, [
        ("Unweighted MLP", "Unweighted MLP"),
        ("Class-Weighted MLP", "Class-Weighted MLP"),
        ("Frank-Hall Ordinal", "Frank-Hall Ordinal")
    ]):
        cm_data = np.array(confusion_matrices_all[cm_key])
        cm_norm = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{m_name}\nNormalized Confusion Matrix", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted PHQ-9 Class")
        ax.set_ylabel("True PHQ-9 Class")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices_comparison.png"), dpi=300)
    plt.close()
    
    # Figure 2: Per-Class F1 Score Comparison Across Key Approaches
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_records = []
    for m_label in ["Logistic Regression", "Unweighted MLP", "Class-Weighted MLP", "XGBoost", "Frank-Hall Ordinal"]:
        p_dict = per_class_all[m_label]
        for c_idx in range(5):
            f1_records.append({
                "Model": m_label,
                "Class": f"Class {c_idx} ({CLASS_NAMES[c_idx]})",
                "F1-Score": p_dict[str(c_idx)]
            })
    f1_df = pd.DataFrame(f1_records)
    sns.barplot(data=f1_df, x="Class", y="F1-Score", hue="Model", palette="tab10", ax=ax)
    ax.set_title("Per-Class F1 Score Across PHQ-9 Depression Severity Classes", fontsize=14, fontweight='bold')
    ax.set_ylim(0.0, 0.7)
    ax.set_ylabel("F1 Score")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_class_f1_comparison.png"), dpi=300)
    plt.close()
    
    # Figure 3: Feature Group Ablations Bar Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ablation_melted = pd.melt(ablation_df, id_vars=["Feature Group", "N_Features"],
                              value_vars=["Test Macro-F1", "Test Bal-Acc"],
                              var_name="Metric", value_name="Score")
    sns.barplot(data=ablation_melted, x="Feature Group", y="Score", hue="Metric", palette="Set2", ax=ax)
    ax.set_title("Feature-Group Ablation Performance on Test Set", fontsize=13, fontweight='bold')
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 0.4)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_group_ablation.png"), dpi=300)
    plt.close()
    
    # Figure 4: Longitudinal Matched Population Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    matched_sub = matched_df[matched_df["Evaluation Population"] == "Matched Cohort (history_len >= 1)"]
    m_melted = pd.melt(matched_sub, id_vars=["Model"], value_vars=["Macro-F1", "Balanced Accuracy", "Accuracy"],
                       var_name="Metric", value_name="Score")
    sns.barplot(data=m_melted, x="Model", y="Score", hue="Metric", palette="mako", ax=ax)
    ax.set_title("Longitudinal Model Comparison on Matched Population (history_len >= 1)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 0.55)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "longitudinal_matched_comparison.png"), dpi=300)
    plt.close()
    
    # Figure 5: High-Resolution t-SNE of Learned Latent Space (from Clean MLP)
    print("Generating t-SNE projection of learned latent representations...")
    model_e2.eval()
    with torch.no_grad():
        _, _, lat_test = model_e2(torch.tensor(X_test, dtype=torch.float32))
    lat_np = lat_test.numpy()
    
    # Sample 1000 points deterministically for clean visualization
    np.random.seed(42)
    sample_idx = np.random.choice(len(lat_np), size=min(1000, len(lat_np)), replace=False)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42)
    embedded_2d = tsne.fit_transform(lat_np[sample_idx])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sample_labels = [CLASS_NAMES[y_test[i]] for i in sample_idx]
    palette = sns.color_palette("viridis", 5)
    sns.scatterplot(
        x=embedded_2d[:, 0], y=embedded_2d[:, 1],
        hue=sample_labels, hue_order=CLASS_NAMES,
        palette=palette, alpha=0.75, s=40, ax=ax
    )
    ax.set_title("t-SNE Projection of Test Set Latent Representations (Clean MLP)", fontsize=13, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title="True PHQ-9 Class", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tsne_latent_space.png"), dpi=300)
    plt.close()
    
    # Figure 6: SHAP Feature Importance on XGBoost
    print("Generating SHAP feature attributions on XGBoost...")
    try:
        # Background sample of 100 points from train
        bg_sample = X_train[np.random.choice(len(X_train), size=100, replace=False)]
        explainer = shap.TreeExplainer(res_xgb_unw["model"], data=bg_sample)
        test_shap_sample = X_test[:150]
        shap_values = explainer.shap_values(test_shap_sample)
        
        plt.figure(figsize=(10, 7))
        # shap_values can be list of arrays for multiclass or 3D array
        if isinstance(shap_values, list):
            # Plot summary for severe depression (Class 4) or overall
            shap.summary_plot(shap_values[4], test_shap_sample, feature_names=feature_cols, max_display=15, show=False)
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap.summary_plot(shap_values[:, :, 4], test_shap_sample, feature_names=feature_cols, max_display=15, show=False)
        else:
            shap.summary_plot(shap_values, test_shap_sample, feature_names=feature_cols, max_display=15, show=False)
            
        plt.title("SHAP Feature Importance (Class 4: Severe Depression)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"), dpi=300)
        plt.close()
        print("SHAP plot generated successfully!")
    except Exception as e:
        print(f"SHAP generation note: {e}")
        
    elapsed = time.time() - start_time
    print(f"\n[DONE] ALL STAGE 3 BENCHMARKS & ANALYSES COMPLETED IN {elapsed:.1f}s!")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Plots saved to:   {PLOTS_DIR}")

if __name__ == "__main__":
    main()
