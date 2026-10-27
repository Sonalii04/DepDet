import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, mean_absolute_error,
    root_mean_squared_error, r2_score, cohen_kappa_score,
    classification_report, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def compute_multiclass_g_mean(recalls_or_y_true, y_pred=None, labels=None) -> float:
    """
    Computes the Multiclass Geometric Mean (G-Mean) across all class recalls:
        G-Mean = (prod_{c=0}^{C-1} Recall_c) ** (1 / C)
    
    Documented Zero-Recall Convention:
    If any class recall is 0.0, the geometric product is 0.0, yielding G-Mean = 0.0.
    This strictly penalizes models that starve minority severity classes.
    """
    if y_pred is not None:
        y_true_arr = np.asarray(recalls_or_y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        if labels is None:
            labels = [0, 1, 2, 3, 4]
        rec_arr = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    else:
        rec_arr = np.asarray(recalls_or_y_true, dtype=np.float64)
        
    if len(rec_arr) == 0:
        return 0.0
    if np.any(rec_arr <= 0.0):
        return 0.0
    # Use log-sum-exp formulation for numerical stability
    return float(np.exp(np.mean(np.log(rec_arr))))

def evaluate_classification(y_true, y_pred, labels=None, y_prob=None) -> dict:
    """
    Computes standardized classification and ordinal metrics for 5-class depression severity.
    
    Parameters:
    - y_true: Ground truth category labels (0 to 4).
    - y_pred: Predicted category labels (0 to 4).
    - labels: List of class labels. Default is [0, 1, 2, 3, 4] for PHQ-9 severity categories.
    - y_prob: Optional predicted class probability matrix of shape (N, C).
    
    Returns:
    Dictionary containing:
    - Standard: accuracy, balanced_accuracy, precision_macro, precision_weighted,
      recall_macro, recall_weighted, macro_f1, weighted_f1, confusion_matrix.
    - Per-class: per_class_precision, per_class_recall, per_class_f1 (keyed by class int and str).
    - Imbalance / Multiclass G-Mean: g_mean, multiclass_g_mean (zero if any class recall is 0.0).
    - Ordinal metrics: quadratic_weighted_kappa, ordinal_class_mae.
    - Probability metrics (if y_prob provided): roc_auc_ovr_macro, pr_auc_ovr_macro, ovr_roc_auc_macro, ovr_pr_auc_macro.
    - Full classification_report dictionary.
    """
    if labels is None:
        labels = [0, 1, 2, 3, 4]
        
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    
    # Standard aggregate metrics
    acc = accuracy_score(y_true_arr, y_pred_arr)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    prec_macro = precision_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true_arr, y_pred_arr, labels=labels, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true_arr, y_pred_arr, labels=labels, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, labels=labels, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, labels=labels, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    
    # Per-class metrics
    prec_per_class = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    
    per_class_precision = {int(c): float(p) for c, p in zip(labels, prec_per_class)}
    per_class_precision.update({str(c): float(p) for c, p in zip(labels, prec_per_class)})
    per_class_recall = {int(c): float(r) for c, r in zip(labels, rec_per_class)}
    per_class_recall.update({str(c): float(r) for c, r in zip(labels, rec_per_class)})
    per_class_f1 = {int(c): float(f) for c, f in zip(labels, f1_per_class)}
    per_class_f1.update({str(c): float(f) for c, f in zip(labels, f1_per_class)})
    
    # Multiclass G-Mean
    g_mean = compute_multiclass_g_mean(rec_per_class)
    
    # Ordinal Metrics
    try:
        qwk = float(cohen_kappa_score(y_true_arr, y_pred_arr, labels=labels, weights='quadratic'))
        if np.isnan(qwk):
            qwk = 0.0
    except Exception:
        qwk = 0.0
        
    ordinal_class_mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    
    # One-vs-Rest ROC-AUC and PR-AUC
    roc_auc_ovr = None
    pr_auc_ovr = None
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == len(labels):
            # ROC-AUC OVR Macro
            try:
                # Check that all classes are present in y_true_arr
                unique_true = set(y_true_arr)
                if len(unique_true) >= 2:
                    roc_auc_ovr = float(roc_auc_score(
                        y_true_arr, y_prob_arr,
                        multi_class='ovr', average='macro', labels=labels
                    ))
            except Exception:
                roc_auc_ovr = None
                
            # PR-AUC OVR Macro
            try:
                pr_aucs = []
                for idx, c in enumerate(labels):
                    y_bin = (y_true_arr == c).astype(int)
                    if len(np.unique(y_bin)) > 1:
                        score_c = average_precision_score(y_bin, y_prob_arr[:, idx])
                        if not np.isnan(score_c):
                            pr_aucs.append(float(score_c))
                pr_auc_ovr = float(np.mean(pr_aucs)) if len(pr_aucs) > 0 else None
            except Exception:
                pr_auc_ovr = None
                
    # Full classification report dictionary
    class_report_dict = classification_report(
        y_true_arr, y_pred_arr, labels=labels, output_dict=True, zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "g_mean": float(g_mean),
        "multiclass_g_mean": float(g_mean),
        "quadratic_weighted_kappa": float(qwk),
        "ordinal_class_mae": float(ordinal_class_mae),
        "roc_auc_ovr_macro": roc_auc_ovr,
        "ovr_roc_auc_macro": roc_auc_ovr,
        "pr_auc_ovr_macro": pr_auc_ovr,
        "ovr_pr_auc_macro": pr_auc_ovr,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report_dict
    }

def evaluate_regression(y_true, y_pred):
    """
    Computes all standard regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }

def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Plots and saves confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_metrics_to_json(metrics_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics to {save_path}")
