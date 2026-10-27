import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, mean_absolute_error,
    root_mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def evaluate_classification(y_true, y_pred, labels=None):
    """
    Computes all standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # pyright: ignore[reportArgumentType]
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # pyright: ignore[reportArgumentType]
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # pyright: ignore[reportArgumentType]
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist()
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
