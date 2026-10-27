"""
Stage 5: Severe-Class Decision Threshold Optimization (WP4).

Implements:
1. Validation-only threshold optimization for Severe Depression (Class 4).
2. Transparent multiclass arbitration rule:
       if p_{i, 4} >= T_4: predict Class 4
       else: predict argmax_{c in {0, 1, 2, 3}} p_{i, c}
3. Clinical utility objective function:
       J(T_4) = Macro-F1_val + 0.50 * F1_val^{(4)}
       subject to Recall_val >= 0.20 and Precision_val >= 0.10
4. Freeze selected threshold T_4* and evaluate once on held-out test set.
5. Machine-readable export of sweep curve, guardrails, and alert trade-offs.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, accuracy_score


def apply_severe_threshold_multiclass(probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies severe-class threshold arbitration to probability matrix of shape (N, 5).
    Guarantees every prediction is a valid integer in [0, 4].
    """
    preds = np.zeros(len(probs), dtype=int)
    # Sub-argmax over non-severe classes [0, 1, 2, 3]
    non_severe_probs = probs[:, :4]
    non_severe_argmax = np.argmax(non_severe_probs, axis=1)
    
    for i in range(len(probs)):
        if probs[i, 4] >= threshold:
            preds[i] = 4
        else:
            preds[i] = non_severe_argmax[i]
            
    return preds


def evaluate_threshold_performance(y_true: np.ndarray, preds: np.ndarray) -> dict:
    """
    Computes severe class (Class 4) and overall multiclass performance for a set of predictions.
    """
    labels = [0, 1, 2, 3, 4]
    macro_f1 = float(f1_score(y_true, preds, labels=labels, average='macro', zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    acc = float(accuracy_score(y_true, preds))
    
    # Severe class metrics
    sev_true = (y_true == 4)
    sev_pred = (preds == 4)
    
    total_severe_true = int(np.sum(sev_true))
    total_alerts = int(np.sum(sev_pred))
    true_positives = int(np.sum(sev_true & sev_pred))
    false_positives = int(np.sum((~sev_true) & sev_pred))
    false_negatives = int(np.sum(sev_true & (~sev_pred)))
    
    sev_recall = float(recall_score(sev_true, sev_pred, zero_division=0))
    sev_precision = float(precision_score(sev_true, sev_pred, zero_division=0))
    sev_f1 = float(f1_score(sev_true, sev_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, preds, labels=labels).tolist()
    
    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_acc": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "severe_metrics": {
            "recall": round(sev_recall, 4),
            "precision": round(sev_precision, 4),
            "f1": round(sev_f1, 4),
            "support": total_severe_true,
            "total_alerts": total_alerts,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        },
        "confusion_matrix": cm
    }


def optimize_severe_threshold(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    min_recall: float = 0.20,
    min_precision: float = 0.10,
    n_thresholds: int = 49
) -> dict:
    """
    Sweeps candidate severe-class thresholds strictly on validation probabilities.
    Maximizes: J(T_4) = Macro-F1 + 0.50 * Severe_F1 subject to guardrails.
    """
    candidate_thresholds = np.linspace(0.02, 0.50, n_thresholds)
    sweep_results = []
    
    best_t = None
    best_j = -1.0
    best_f1_sev = -1.0
    guardrails_met = False
    
    # Also evaluate reference default argmax
    default_val_preds = np.argmax(val_probs, axis=1)
    default_eval = evaluate_threshold_performance(y_val, default_val_preds)
    
    for t in candidate_thresholds:
        preds = apply_severe_threshold_multiclass(val_probs, float(t))
        metrics = evaluate_threshold_performance(y_val, preds)
        
        j_val = metrics["macro_f1"] + 0.50 * metrics["severe_metrics"]["f1"]
        rec = metrics["severe_metrics"]["recall"]
        prec = metrics["severe_metrics"]["precision"]
        
        meets_guardrails = (rec >= min_recall) and (prec >= min_precision)
        
        record = {
            "threshold": round(float(t), 4),
            "j_objective": round(j_val, 4),
            "macro_f1": metrics["macro_f1"],
            "balanced_acc": metrics["balanced_acc"],
            "accuracy": metrics["accuracy"],
            "severe_recall": metrics["severe_metrics"]["recall"],
            "severe_precision": metrics["severe_metrics"]["precision"],
            "severe_f1": metrics["severe_metrics"]["f1"],
            "total_alerts": metrics["severe_metrics"]["total_alerts"],
            "false_positives": metrics["severe_metrics"]["false_positives"],
            "false_negatives": metrics["severe_metrics"]["false_negatives"],
            "meets_guardrails": meets_guardrails
        }
        sweep_results.append(record)
        
        if meets_guardrails:
            guardrails_met = True
            # Prioritize J objective, tie-break by Severe F1
            if (j_val > best_j) or (np.isclose(j_val, best_j) and metrics["severe_metrics"]["f1"] > best_f1_sev):
                best_j = j_val
                best_f1_sev = metrics["severe_metrics"]["f1"]
                best_t = float(t)
                
    # Fallback if no threshold met both guardrails
    fallback_used = False
    if best_t is None:
        fallback_used = True
        best_record = max(sweep_results, key=lambda r: r["j_objective"])
        best_t = best_record["threshold"]
        best_j = best_record["j_objective"]
        
    # Re-evaluate chosen threshold on validation
    optimal_val_preds = apply_severe_threshold_multiclass(val_probs, best_t)
    optimal_val_eval = evaluate_threshold_performance(y_val, optimal_val_preds)
    
    return {
        "selected_threshold": round(best_t, 4),
        "validation_objective_j": round(best_j, 4),
        "guardrails": {
            "min_recall": min_recall,
            "min_precision": min_precision,
            "guardrails_met_on_validation": guardrails_met,
            "fallback_unconstrained_used": fallback_used
        },
        "default_argmax_validation": default_eval,
        "calibrated_threshold_validation": optimal_val_eval,
        "sweep_curve": sweep_results
    }


def evaluate_frozen_threshold_on_test(
    test_probs: np.ndarray,
    y_test: np.ndarray,
    selected_threshold: float
) -> dict:
    """
    Evaluates the frozen validation-selected threshold ONCE on the held-out test set.
    """
    default_test_preds = np.argmax(test_probs, axis=1)
    default_eval = evaluate_threshold_performance(y_test, default_test_preds)
    
    calibrated_test_preds = apply_severe_threshold_multiclass(test_probs, selected_threshold)
    calibrated_eval = evaluate_threshold_performance(y_test, calibrated_test_preds)
    
    return {
        "frozen_threshold": selected_threshold,
        "default_argmax_test": default_eval,
        "calibrated_threshold_test": calibrated_eval,
        "test_predictions": calibrated_test_preds.tolist()
    }
