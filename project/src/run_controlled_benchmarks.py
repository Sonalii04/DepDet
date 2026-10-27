import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import copy

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_dataset, inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from evaluation import evaluate_classification, evaluate_regression, save_metrics_to_json, plot_confusion_matrix
from baseline import ClassificationMLP, train_baseline_mlp, prepare_dataset_arrays, compute_class_weights
from longitudinal import train_longitudinal_experiment
from oversampling import latent_space_smote

def run_experiment_1_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, class_names):
    """
    Experiment 1: Regularized Multiclass Logistic Regression (L2 penalty)
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 1: Regularized Logistic Regression (L2)")
    print("=" * 60)
    clf = LogisticRegression(
        penalty='l2',
        C=0.1,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    clf.fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)
    
    val_metrics = evaluate_classification(y_val, val_preds)
    test_metrics = evaluate_classification(y_test, test_preds)
    
    print(f"Logistic Regression -> Val Macro-F1: {val_metrics['macro_f1']:.4f}, Val Bal-Acc: {val_metrics['balanced_accuracy']:.4f}")
    return val_metrics, test_metrics, clf

def run_experiment_2_classification_mlp(train_df, val_df, test_df, f_cols):
    """
    Experiment 2: Corrected Clean Classification-Only MLP
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 2: Clean Classification-Only MLP (Standard Cross-Entropy)")
    print("=" * 60)
    model, prep, metrics, _, _ = train_baseline_mlp(
        train_df, val_df, test_df, f_cols,
        use_class_weights=False,
        aux_reg_weight=0.0,
        epochs=30,
        patience=7,
        seed=42,
        checkpoint_name="exp2_classification_mlp_best.pt"
    )
    val_m = metrics["val_classification"]
    test_m = metrics["test_classification"]
    print(f"Classification MLP -> Val Macro-F1: {val_m['macro_f1']:.4f}, Val Bal-Acc: {val_m['balanced_accuracy']:.4f}")
    return val_m, test_m, model

def run_experiment_3_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Experiment 3: Strong Tabular Model (HistGradientBoostingClassifier)
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 3: Strong Tabular Model (HistGradientBoostingClassifier)")
    print("=" * 60)
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    hgb.fit(X_train, y_train)
    val_preds = hgb.predict(X_val)
    test_preds = hgb.predict(X_test)
    
    val_metrics = evaluate_classification(y_val, val_preds)
    test_metrics = evaluate_classification(y_test, test_preds)
    
    print(f"HistGradientBoosting -> Val Macro-F1: {val_metrics['macro_f1']:.4f}, Val Bal-Acc: {val_metrics['balanced_accuracy']:.4f}")
    return val_metrics, test_metrics, hgb

def run_experiment_4_imbalance_strategies(train_df, val_df, test_df, f_cols, X_train, y_train, X_val, y_val, X_test, y_test, unweighted_encoder=None):
    """
    Experiment 4: Class-Imbalance Handling
    4a. Class-Weighted Cross-Entropy
    4b. Latent SMOTE Oversampling (extracted from trained unweighted MLP encoder)
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 4: Class-Imbalance Handling Strategies")
    print("=" * 60)
    # 4a. Class-Weighted Cross-Entropy
    print(">>> 4a. Class-Weighted Cross-Entropy...")
    m_weighted, _, res_weighted, _, _ = train_baseline_mlp(
        train_df, val_df, test_df, f_cols,
        use_class_weights=True,
        aux_reg_weight=0.0,
        epochs=30,
        patience=7,
        seed=42,
        checkpoint_name="exp4a_class_weighted_mlp_best.pt"
    )
    val_weighted = res_weighted["val_classification"]
    test_weighted = res_weighted["test_classification"]
    print(f"Class-Weighted MLP -> Val Macro-F1: {val_weighted['macro_f1']:.4f}, Val Bal-Acc: {val_weighted['balanced_accuracy']:.4f}")
    
    # 4b. Latent SMOTE Oversampling
    print("\n>>> 4b. Latent-Space SMOTE Oversampling on TRAIN...")
    # Extract latent vectors strictly from trained unweighted MLP encoder
    if unweighted_encoder is None:
        unweighted_encoder, _, _, _, _ = train_baseline_mlp(
            train_df, val_df, test_df, f_cols,
            use_class_weights=False,
            aux_reg_weight=0.0,
            epochs=30,
            patience=7,
            seed=42,
            checkpoint_name="smote_unweighted_encoder_best.pt"
        )
    unweighted_encoder.eval()
    with torch.no_grad():
        _, _, lat_tr = unweighted_encoder(torch.tensor(X_train, dtype=torch.float32))
        _, _, lat_va = unweighted_encoder(torch.tensor(X_val, dtype=torch.float32))
        _, _, lat_te = unweighted_encoder(torch.tensor(X_test, dtype=torch.float32))
        
    lat_tr_np = lat_tr.numpy()
    lat_va_np = lat_va.numpy()
    lat_te_np = lat_te.numpy()
    
    # Apply SMOTE strictly to training latents
    bal_lat_tr, bal_y_tr = latent_space_smote(lat_tr_np, y_train)
    
    # Train classifier head on balanced latents
    clf_head = nn.Linear(bal_lat_tr.shape[1], 5)
    opt = torch.optim.Adam(clf_head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    X_smote_t = torch.tensor(bal_lat_tr, dtype=torch.float32)
    y_smote_t = torch.tensor(bal_y_tr, dtype=torch.long)
    
    best_smote_f1 = -1.0
    best_smote_bal = -1.0
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
            v_out = clf_head(torch.tensor(lat_va_np, dtype=torch.float32))
            v_pred = torch.argmax(v_out, dim=1).numpy()
            v_cls = evaluate_classification(y_val, v_pred)
            if v_cls["macro_f1"] > best_smote_f1:
                best_smote_f1 = v_cls["macro_f1"]
                best_smote_bal = v_cls["balanced_accuracy"]
                best_smote_weights = copy.deepcopy(clf_head.state_dict())
                
    if best_smote_weights is not None:
        clf_head.load_state_dict(best_smote_weights)
        
    clf_head.eval()
    with torch.no_grad():
        v_final = clf_head(torch.tensor(lat_va_np, dtype=torch.float32))
        val_preds_smote = torch.argmax(v_final, dim=1).numpy()
        t_final = clf_head(torch.tensor(lat_te_np, dtype=torch.float32))
        test_preds_smote = torch.argmax(t_final, dim=1).numpy()
        
    val_smote = evaluate_classification(y_val, val_preds_smote)
    test_smote = evaluate_classification(y_test, test_preds_smote)
    print(f"Latent SMOTE -> Val Macro-F1: {val_smote['macro_f1']:.4f}, Val Bal-Acc: {val_smote['balanced_accuracy']:.4f}")
    
    return (val_weighted, test_weighted), (val_smote, test_smote)

def run_experiment_5_ordinal_classification(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Experiment 5: Ordinal Classification via Frank & Hall Binary Decomposition
    Decomposes 5 ordered classes into 4 binary classifiers:
    Classifier k predicts P(Y > k) for k in {0, 1, 2, 3}.
    Then:
    P(Y = 0) = 1 - P(Y > 0)
    P(Y = 1) = P(Y > 0) - P(Y > 1)
    P(Y = 2) = P(Y > 1) - P(Y > 2)
    P(Y = 3) = P(Y > 2) - P(Y > 3)
    P(Y = 4) = P(Y > 3)
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 5: Ordinal Classification (Frank & Hall Decomposition)")
    print("=" * 60)
    
    models = []
    for k in range(4):
        # Binary target: 1 if Y > k else 0
        y_tr_bin = (y_train > k).astype(int)
        clf = LogisticRegression(penalty='l2', C=0.1, max_iter=1000, random_state=42 + k)
        clf.fit(X_train, y_tr_bin)
        models.append(clf)
        
    def predict_ordinal_probs(models, X):
        # Get P(Y > k) for each threshold
        p_greater = np.zeros((len(X), 4))
        for k in range(4):
            # proba[:, 1] is P(Y > k)
            p_greater[:, k] = models[k].predict_proba(X)[:, 1]
            
        # Ensure monotonic probabilities P(Y > 0) >= P(Y > 1) >= P(Y > 2) >= P(Y > 3)
        for k in range(1, 4):
            p_greater[:, k] = np.minimum(p_greater[:, k], p_greater[:, k - 1])
            
        probs = np.zeros((len(X), 5))
        probs[:, 0] = 1.0 - p_greater[:, 0]
        probs[:, 1] = p_greater[:, 0] - p_greater[:, 1]
        probs[:, 2] = p_greater[:, 1] - p_greater[:, 2]
        probs[:, 3] = p_greater[:, 2] - p_greater[:, 3]
        probs[:, 4] = p_greater[:, 3]
        probs = np.clip(probs, 0.0, 1.0)
        # Normalize rows
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
        
    val_probs = predict_ordinal_probs(models, X_val)
    test_probs = predict_ordinal_probs(models, X_test)
    
    val_preds = np.argmax(val_probs, axis=1)
    test_preds = np.argmax(test_probs, axis=1)
    
    val_metrics = evaluate_classification(y_val, val_preds)
    test_metrics = evaluate_classification(y_test, test_preds)
    
    print(f"Ordinal Decomposition -> Val Macro-F1: {val_metrics['macro_f1']:.4f}, Val Bal-Acc: {val_metrics['balanced_accuracy']:.4f}")
    return val_metrics, test_metrics, models

def run_experiment_6_longitudinal_comparisons(train_df, val_df, test_df, f_cols):
    """
    Experiment 6: Longitudinal History Comparisons on Matched Population
    6a. Current Only
    6b. History Only (strictly valid previous observations)
    6c. Current + History Fusion (with proper masking and learned no-history representation)
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 6: Longitudinal Temporal Modeling")
    print("=" * 60)
    
    res_curr = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode="current_only", epochs=25, seed=42)
    print(f"6a. Current Only -> Val Macro-F1: {res_curr['val_classification']['macro_f1']:.4f}, Val Bal-Acc: {res_curr['val_classification']['balanced_accuracy']:.4f}")
    
    res_hist = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode="history_only", epochs=25, seed=42)
    print(f"6b. History Only -> Val Macro-F1: {res_hist['val_classification']['macro_f1']:.4f}, Val Bal-Acc: {res_hist['val_classification']['balanced_accuracy']:.4f}")
    
    res_fuse = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode="current_plus_history", epochs=25, seed=42)
    print(f"6c. Current + History Fusion -> Val Macro-F1: {res_fuse['val_classification']['macro_f1']:.4f}, Val Bal-Acc: {res_fuse['val_classification']['balanced_accuracy']:.4f}")
    
    return res_curr, res_hist, res_fuse

def run_experiment_7_auxiliary_regression_ablation(train_df, val_df, test_df, f_cols):
    """
    Experiment 7: Controlled Auxiliary Regression Ablation
    Tests whether auxiliary PHQ-9 score regression benefits classification across weights:
    lambda in [0.0, 0.1, 0.2, 0.5] evaluated strictly on VALIDATION data.
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT 7: Auxiliary Regression Multi-Task Ablation")
    print("=" * 60)
    
    weights = [0.0, 0.1, 0.2, 0.5]
    results = {}
    
    for w in weights:
        print(f">>> Testing aux_reg_weight = {w}...")
        _, _, metrics, _, _ = train_baseline_mlp(
            train_df, val_df, test_df, f_cols,
            use_class_weights=False,
            aux_reg_weight=w,
            epochs=25,
            patience=6,
            seed=42,
            checkpoint_name=f"exp7_aux_reg_w{w}.pt"
        )
        val_m = metrics["val_classification"]
        results[w] = {
            "val_macro_f1": val_m["macro_f1"],
            "val_bal_acc": val_m["balanced_accuracy"],
            "val_accuracy": val_m["accuracy"],
            "test_macro_f1": metrics["test_classification"]["macro_f1"],
            "test_balanced_acc": metrics["test_classification"]["balanced_accuracy"],
            "test_accuracy": metrics["test_classification"]["accuracy"],
            "test_mae": metrics["test_regression"]["mae"] if metrics["test_regression"] else None
        }
        print(f"Weight {w} -> Val Macro-F1: {val_m['macro_f1']:.4f}, Val Bal-Acc: {val_m['balanced_accuracy']:.4f}")
        
    return results

def main():
    print("=" * 80)
    print("STARTING CONTROLLED EXPERIMENTAL BENCHMARK PIPELINE")
    print("=" * 80)
    
    parquet_path = "d:/projects/DepDet/anon_processed_df_parquet"
    df = load_raw_dataset(parquet_path)
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    
    train_df, val_df, test_df, split_info = perform_subject_level_split(df, seed=42)
    
    prep = DataPreprocessor(feature_cols=f_cols)
    X_train, y_train, y_train_score, _, _ = prepare_dataset_arrays(train_df, prep, fit_preprocessor=True)
    X_val, y_val, y_val_score, _, _ = prepare_dataset_arrays(val_df, prep, fit_preprocessor=False)
    X_test, y_test, y_test_score, _, _ = prepare_dataset_arrays(test_df, prep, fit_preprocessor=False)
    
    class_names = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]
    
    summary_table = []
    
    # Exp 1: Logistic Regression
    val_e1, test_e1, _ = run_experiment_1_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, class_names)
    summary_table.append({
        "Experiment": "Exp 1: Logistic Regression (L2)",
        "Model": "LogisticRegression (C=0.1)",
        "Features": "150 Concurrent Features",
        "Imbalance Strategy": "None (Natural)",
        "Val Macro-F1": val_e1["macro_f1"],
        "Val Bal-Acc": val_e1["balanced_accuracy"],
        "Val Accuracy": val_e1["accuracy"],
        "Test Macro-F1": test_e1["macro_f1"],
        "Test Bal-Acc": test_e1["balanced_accuracy"],
        "Test Accuracy": test_e1["accuracy"]
    })
    
    # Exp 2: Corrected Classification-Only MLP
    val_e2, test_e2, model_e2 = run_experiment_2_classification_mlp(train_df, val_df, test_df, f_cols)
    summary_table.append({
        "Experiment": "Exp 2: Classification MLP",
        "Model": "ClassificationMLP (2-layer)",
        "Features": "150 Concurrent Features",
        "Imbalance Strategy": "None (Natural)",
        "Val Macro-F1": val_e2["macro_f1"],
        "Val Bal-Acc": val_e2["balanced_accuracy"],
        "Val Accuracy": val_e2["accuracy"],
        "Test Macro-F1": test_e2["macro_f1"],
        "Test Bal-Acc": test_e2["balanced_accuracy"],
        "Test Accuracy": test_e2["accuracy"]
    })
    
    # Exp 3: Strong Tabular Model (HistGradientBoosting)
    val_e3, test_e3, _ = run_experiment_3_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test)
    summary_table.append({
        "Experiment": "Exp 3: HistGradientBoosting",
        "Model": "HistGradientBoostingClassifier",
        "Features": "150 Concurrent Features",
        "Imbalance Strategy": "None (Natural)",
        "Val Macro-F1": val_e3["macro_f1"],
        "Val Bal-Acc": val_e3["balanced_accuracy"],
        "Val Accuracy": val_e3["accuracy"],
        "Test Macro-F1": test_e3["macro_f1"],
        "Test Bal-Acc": test_e3["balanced_accuracy"],
        "Test Accuracy": test_e3["accuracy"]
    })
    
    # Exp 4: Imbalance Strategies (Passing trained clean unweighted model_e2 as encoder for SMOTE)
    (val_4a, test_4a), (val_4b, test_4b) = run_experiment_4_imbalance_strategies(
        train_df, val_df, test_df, f_cols, X_train, y_train, X_val, y_val, X_test, y_test,
        unweighted_encoder=model_e2
    )
    summary_table.append({
        "Experiment": "Exp 4a: Class-Weighted MLP",
        "Model": "ClassificationMLP (Weighted CE)",
        "Features": "150 Concurrent Features",
        "Imbalance Strategy": "Inverse Frequency Weights",
        "Val Macro-F1": val_4a["macro_f1"],
        "Val Bal-Acc": val_4a["balanced_accuracy"],
        "Val Accuracy": val_4a["accuracy"],
        "Test Macro-F1": test_4a["macro_f1"],
        "Test Bal-Acc": test_4a["balanced_accuracy"],
        "Test Accuracy": test_4a["accuracy"]
    })
    summary_table.append({
        "Experiment": "Exp 4b: Latent SMOTE",
        "Model": "MLP + Latent SMOTE Head",
        "Features": "150 Concurrent Features",
        "Imbalance Strategy": "Latent SMOTE (Train Only)",
        "Val Macro-F1": val_4b["macro_f1"],
        "Val Bal-Acc": val_4b["balanced_accuracy"],
        "Val Accuracy": val_4b["accuracy"],
        "Test Macro-F1": test_4b["macro_f1"],
        "Test Bal-Acc": test_4b["balanced_accuracy"],
        "Test Accuracy": test_4b["accuracy"]
    })
    
    # Exp 5: Ordinal Classification
    val_e5, test_e5, _ = run_experiment_5_ordinal_classification(X_train, y_train, X_val, y_val, X_test, y_test)
    summary_table.append({
        "Experiment": "Exp 5: Ordinal Decomposition",
        "Model": "Frank & Hall Ordinal LR",
        "Features": "150 Concurrent Features",
        "Imbalance Strategy": "Ordered Cumulative Logits",
        "Val Macro-F1": val_e5["macro_f1"],
        "Val Bal-Acc": val_e5["balanced_accuracy"],
        "Val Accuracy": val_e5["accuracy"],
        "Test Macro-F1": test_e5["macro_f1"],
        "Test Bal-Acc": test_e5["balanced_accuracy"],
        "Test Accuracy": test_e5["accuracy"]
    })
    
    # Exp 6: Longitudinal Models
    res_curr, res_hist, res_fuse = run_experiment_6_longitudinal_comparisons(train_df, val_df, test_df, f_cols)
    summary_table.append({
        "Experiment": "Exp 6a: Longitudinal Current Only",
        "Model": "LongitudinalGRU (Current Mode)",
        "Features": "Current Assessment Xt",
        "Imbalance Strategy": "None",
        "Val Macro-F1": res_curr["val_classification"]["macro_f1"],
        "Val Bal-Acc": res_curr["val_classification"]["balanced_accuracy"],
        "Val Accuracy": res_curr["val_classification"]["accuracy"],
        "Test Macro-F1": res_curr["test_classification"]["macro_f1"],
        "Test Bal-Acc": res_curr["test_classification"]["balanced_accuracy"],
        "Test Accuracy": res_curr["test_classification"]["accuracy"]
    })
    summary_table.append({
        "Experiment": "Exp 6b: Longitudinal History Only",
        "Model": "LongitudinalGRU (History Mode)",
        "Features": "Valid Past Sequence X1:t-1",
        "Imbalance Strategy": "None",
        "Val Macro-F1": res_hist["val_classification"]["macro_f1"],
        "Val Bal-Acc": res_hist["val_classification"]["balanced_accuracy"],
        "Val Accuracy": res_hist["val_classification"]["accuracy"],
        "Test Macro-F1": res_hist["test_classification"]["macro_f1"],
        "Test Bal-Acc": res_hist["test_classification"]["balanced_accuracy"],
        "Test Accuracy": res_hist["test_classification"]["accuracy"]
    })
    summary_table.append({
        "Experiment": "Exp 6c: Longitudinal Fusion",
        "Model": "LongitudinalGRU (Fusion Mode)",
        "Features": "Current Xt + Masked History X1:t-1",
        "Imbalance Strategy": "None",
        "Val Macro-F1": res_fuse["val_classification"]["macro_f1"],
        "Val Bal-Acc": res_fuse["val_classification"]["balanced_accuracy"],
        "Val Accuracy": res_fuse["val_classification"]["accuracy"],
        "Test Macro-F1": res_fuse["test_classification"]["macro_f1"],
        "Test Bal-Acc": res_fuse["test_classification"]["balanced_accuracy"],
        "Test Accuracy": res_fuse["test_classification"]["accuracy"]
    })
    
    # Exp 7: Auxiliary Multi-task Regression Grid
    aux_results = run_experiment_7_auxiliary_regression_ablation(train_df, val_df, test_df, f_cols)
    for w, res_w in aux_results.items():
        summary_table.append({
            "Experiment": f"Exp 7: Auxiliary Reg (w={w})",
            "Model": f"Multi-Task MLP (weight={w})",
            "Features": "150 Concurrent Features",
            "Imbalance Strategy": "None",
            "Val Macro-F1": res_w["val_macro_f1"],
            "Val Bal-Acc": res_w["val_bal_acc"],
            "Val Accuracy": res_w["val_accuracy"],
            "Test Macro-F1": res_w["test_macro_f1"],
            "Test Bal-Acc": res_w["test_balanced_acc"],
            "Test Accuracy": res_w["test_accuracy"]
        })
        
    summary_df = pd.DataFrame(summary_table)
    csv_path = "d:/projects/DepDet/project/results/metrics/controlled_benchmarks_summary.csv"
    json_path = "d:/projects/DepDet/project/results/metrics/controlled_benchmarks_summary.json"
    summary_df.to_csv(csv_path, index=False)
    save_metrics_to_json(summary_table, json_path)
    
    print("\n" + "=" * 80)
    print("CONTROLLED BENCHMARK SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved results to {csv_path} and {json_path}")

if __name__ == "__main__":
    main()
