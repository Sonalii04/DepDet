"""
Stage 5 Master Pipeline Runner:
Executes all Stage 5 work packages (WP2 to WP9) end-to-end,
records all metrics into project/results/stage5_model_enhancements/,
and builds the interactive research dashboard.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path("d:/projects/DepDet")
SRC_DIR = PROJECT_ROOT / "project" / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from project.src.xgboost_model import compute_multiclass_sample_weights
from project.src.stage5_benchmarks import (
    prepare_stage5_datasets,
    train_and_eval_classifier,
    run_probability_calibration,
    run_continuous_score_regression,
)
from project.src.stage5_threshold_tuning import (
    optimize_severe_threshold,
    evaluate_frozen_threshold_on_test,
)
from project.src.stage5_explainability import compute_stage5_treeshap
from project.src.stage5_dashboard_generator import generate_interactive_dashboard


def run_stage5_full_pipeline():
    print("=" * 85)
    print("[STAGE 5] CONTROLLED MODEL ENHANCEMENT & CLINICAL DECISION SUPPORT PIPELINE")
    print("=" * 85)
    
    output_dir = PROJECT_ROOT / "project" / "results" / "stage5_model_enhancements"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Prepare Datasets (Policy A & Policy B)
    # -------------------------------------------------------------------------
    print("\n[Step 1/7] Loading Datasets and Constructing Policy A and Policy B Feature Pools...")
    t0 = time.time()
    data = prepare_stage5_datasets()
    print(f"Data prepared in {time.time()-t0:.2f}s:")
    print(f"  • Train: {data['meta']['train_n']} rows ({data['meta']['train_users_n']} participants)")
    print(f"  • Val:   {data['meta']['val_n']} rows ({data['meta']['val_users_n']} participants)")
    print(f"  • Test:  {data['meta']['test_n']} rows ({data['meta']['test_users_n']} participants)")
    print(f"  • Verified Longitudinal Transitions (Test): {data['meta']['test_transitions_n']} transitions")
    
    # Compute class weights for training
    y_train = data["Policy_A"]["y_train"]
    weights_train = compute_multiclass_sample_weights(y_train, num_classes=5)
    
    # -------------------------------------------------------------------------
    # 2. Train Tabular Benchmark Classifiers (WP2 & WP3)
    # -------------------------------------------------------------------------
    print("\n[Step 2/7] Training & Evaluating Tabular Benchmarks (Policy A: Sensor-Only)...")
    
    # Model 1: XGBoost (Class-Weighted)
    xgb_weighted_A = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3.0,
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        random_state=42, tree_method="hist", n_jobs=-1
    )
    res_xgb_weighted_A = train_and_eval_classifier(
        xgb_weighted_A,
        data["Policy_A"]["X_train"], data["Policy_A"]["y_train"],
        data["Policy_A"]["X_val"], data["Policy_A"]["y_val"],
        data["Policy_A"]["X_test"], data["Policy_A"]["y_test"],
        sample_weight=weights_train
    )
    
    # Model 2: XGBoost (Unweighted)
    xgb_unweighted_A = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3.0,
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        random_state=42, tree_method="hist", n_jobs=-1
    )
    res_xgb_unweighted_A = train_and_eval_classifier(
        xgb_unweighted_A,
        data["Policy_A"]["X_train"], data["Policy_A"]["y_train"],
        data["Policy_A"]["X_val"], data["Policy_A"]["y_val"],
        data["Policy_A"]["X_test"], data["Policy_A"]["y_test"],
        sample_weight=None
    )
    
    # Model 3: HistGradientBoosting (Class-Weighted)
    hgb_weighted_A = HistGradientBoostingClassifier(
        max_iter=150, max_depth=5, learning_rate=0.05, class_weight='balanced', random_state=42
    )
    res_hgb_weighted_A = train_and_eval_classifier(
        hgb_weighted_A,
        data["Policy_A"]["X_train"], data["Policy_A"]["y_train"],
        data["Policy_A"]["X_val"], data["Policy_A"]["y_val"],
        data["Policy_A"]["X_test"], data["Policy_A"]["y_test"],
        sample_weight=None
    )
    
    # Model 4: Majority Class Baseline
    dummy_model = DummyClassifier(strategy="most_frequent")
    res_dummy = train_and_eval_classifier(
        dummy_model,
        data["Policy_A"]["X_train"], data["Policy_A"]["y_train"],
        data["Policy_A"]["X_val"], data["Policy_A"]["y_val"],
        data["Policy_A"]["X_test"], data["Policy_A"]["y_test"],
        sample_weight=None
    )
    
    print("\n[Step 3/7] Training & Evaluating Policy B Models (Prior-PHQ-9-Assisted)...")
    # Model 5: XGBoost (Policy B: Sensor + Prior PHQ-9)
    xgb_weighted_B = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3.0,
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        random_state=42, tree_method="hist", n_jobs=-1
    )
    res_xgb_weighted_B = train_and_eval_classifier(
        xgb_weighted_B,
        data["Policy_B"]["X_train"], data["Policy_B"]["y_train"],
        data["Policy_B"]["X_val"], data["Policy_B"]["y_val"],
        data["Policy_B"]["X_test"], data["Policy_B"]["y_test"],
        sample_weight=weights_train
    )
    
    # Model 6: HistGradientBoosting (Policy B)
    hgb_weighted_B = HistGradientBoostingClassifier(
        max_iter=150, max_depth=5, learning_rate=0.05, class_weight='balanced', random_state=42
    )
    res_hgb_weighted_B = train_and_eval_classifier(
        hgb_weighted_B,
        data["Policy_B"]["X_train"], data["Policy_B"]["y_train"],
        data["Policy_B"]["X_val"], data["Policy_B"]["y_val"],
        data["Policy_B"]["X_test"], data["Policy_B"]["y_test"],
        sample_weight=None
    )
    
    # Also evaluate Policy B on the transition subset (N=973 test rows where prior PHQ-9 strictly exists)
    test_trans_mask = data["test_df"]["has_transition"].values
    y_test_trans = data["Policy_B"]["y_test"][test_trans_mask]
    probs_xgb_B_trans = res_xgb_weighted_B["test_probs"][test_trans_mask]
    preds_xgb_B_trans = np.argmax(probs_xgb_B_trans, axis=1)
    from project.src.evaluation import evaluate_classification
    trans_metrics_xgb_B = evaluate_classification(y_test_trans, preds_xgb_B_trans, y_prob=probs_xgb_B_trans)
    
    # Stage 4 Frozen Reference Models (W=1 GRU & Persistence Baseline)
    gru_ref = {
        "policy": "Policy A (Sensor-Only)",
        "model_name": "Windowed Fusion GRU (W=1, Stage 4 Selected)",
        "val_macro_f1": 0.3359,
        "test_macro_f1": 0.3065,
        "test_bal_acc": 0.3305,
        "test_accuracy": 0.3898,
        "test_qwk": 0.4313,
        "severe_recall": 0.1667,
        "severe_precision": 0.3636,
        "test_score_mae": 4.1130
    }
    persist_ref = {
        "policy": "Clinical Baseline",
        "model_name": "Clinical Persistence Baseline (y_hat_t = y_{t-1})",
        "val_macro_f1": 0.0,
        "test_macro_f1": 0.4810, # evaluated on N=973 transitions
        "test_bal_acc": 0.3012,
        "test_accuracy": 0.5858,
        "test_qwk": 0.4257,
        "severe_recall": 0.0,
        "severe_precision": 0.0,
        "test_score_mae": 2.9353
    }
    
    # Consolidate benchmark rows
    def pack_benchmark_row(res, name, policy, score_mae=None):
        return {
            "policy": policy,
            "model_name": name,
            "val_macro_f1": round(res["val_metrics"]["macro_f1"], 4),
            "test_macro_f1": round(res["test_metrics"]["macro_f1"], 4),
            "test_bal_acc": round(res["test_metrics"]["balanced_accuracy"], 4),
            "test_accuracy": round(res["test_metrics"]["accuracy"], 4),
            "test_qwk": round(res["test_metrics"]["quadratic_weighted_kappa"], 4),
            "severe_recall": round(res["test_metrics"]["per_class_recall"]["4"], 4),
            "severe_precision": round(res["test_metrics"]["per_class_precision"]["4"], 4),
            "test_score_mae": score_mae,
            "fit_time_sec": res["fit_time_sec"]
        }
        
    benchmarks_list = [
        pack_benchmark_row(res_xgb_weighted_A, "XGBoost (Class-Weighted)", "Policy A (Sensor-Only)"),
        pack_benchmark_row(res_xgb_unweighted_A, "XGBoost (Unweighted)", "Policy A (Sensor-Only)"),
        pack_benchmark_row(res_hgb_weighted_A, "HistGradientBoosting (Class-Weighted)", "Policy A (Sensor-Only)"),
        gru_ref,
        pack_benchmark_row(res_dummy, "Majority Class Baseline", "Policy A (Sensor-Only)"),
        pack_benchmark_row(res_xgb_weighted_B, "XGBoost (Class-Weighted + Prior PHQ-9)", "Policy B (Prior-PHQ-9-Assisted)"),
        pack_benchmark_row(res_hgb_weighted_B, "HistGradientBoosting (+ Prior PHQ-9)", "Policy B (Prior-PHQ-9-Assisted)"),
        persist_ref
    ]
    
    benchmarks_df = pd.DataFrame(benchmarks_list)
    benchmarks_df.to_csv(output_dir / "stage5_benchmark_comparison.csv", index=False)
    print("\nBenchmark Evaluation Summary Table:")
    print(benchmarks_df[["policy", "model_name", "val_macro_f1", "test_macro_f1", "test_bal_acc", "test_accuracy", "test_qwk", "severe_recall"]].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 3. Severe-Class Decision Threshold Optimization (WP4)
    # -------------------------------------------------------------------------
    print("\n[Step 4/7] Running Validation-Only Severe-Class Decision Threshold Sweep on XGBoost Policy A...")
    threshold_results = optimize_severe_threshold(
        res_xgb_weighted_A["val_probs"],
        data["Policy_A"]["y_val"],
        min_recall=0.20,
        min_precision=0.10,
        n_thresholds=49
    )
    
    frozen_t = threshold_results["selected_threshold"]
    print(f"  -> Validation Optimal Threshold Selected: T_4* = {frozen_t:.4f} (J = {threshold_results['validation_objective_j']:.4f})")
    print(f"  -> Validation Severe Recall: {threshold_results['calibrated_threshold_validation']['severe_metrics']['recall']*100:.1f}%, Precision: {threshold_results['calibrated_threshold_validation']['severe_metrics']['precision']*100:.1f}%")
    
    # Evaluate frozen threshold once on test set
    test_threshold_eval = evaluate_frozen_threshold_on_test(
        res_xgb_weighted_A["test_probs"],
        data["Policy_A"]["y_test"],
        frozen_t
    )
    
    threshold_full_record = {
        "optimization_procedure": {
            "model_evaluated": "XGBoost (Class-Weighted, Policy A)",
            "selection_split": "Validation Split Only (N=1,597)",
            "selected_threshold": frozen_t,
            "objective_j": threshold_results["validation_objective_j"],
            "guardrails": threshold_results["guardrails"]
        },
        "validation_performance": {
            "default_argmax": threshold_results["default_argmax_validation"],
            "calibrated_threshold": threshold_results["calibrated_threshold_validation"]
        },
        "test_performance_frozen": {
            "default_argmax": test_threshold_eval["default_argmax_test"],
            "calibrated_threshold": test_threshold_eval["calibrated_threshold_test"]
        },
        "sweep_curve": threshold_results["sweep_curve"]
    }
    
    with open(output_dir / "stage5_threshold_optimization.json", "w") as f:
        json.dump(threshold_full_record, f, indent=2)
        
    print(f"  -> Held-Out Test Evaluation with T_4* = {frozen_t:.4f}:")
    t_test_res = test_threshold_eval["calibrated_threshold_test"]["severe_metrics"]
    def_test_res = test_threshold_eval["default_argmax_test"]["severe_metrics"]
    print(f"     • Default Argmax: Severe Recall = {def_test_res['recall']*100:.1f}% ({def_test_res['true_positives']}/{def_test_res['support']}), Precision = {def_test_res['precision']*100:.1f}%, Alerts = {def_test_res['total_alerts']}")
    print(f"     • Calibrated T_4: Severe Recall = {t_test_res['recall']*100:.1f}% ({t_test_res['true_positives']}/{t_test_res['support']}), Precision = {t_test_res['precision']*100:.1f}%, Alerts = {t_test_res['total_alerts']} (FP = {t_test_res['false_positives']})")
    
    # -------------------------------------------------------------------------
    # 4. Probability Calibration & Uncertainty (WP5)
    # -------------------------------------------------------------------------
    print("\n[Step 5/7] Evaluating Probability Calibration (Uncalibrated vs. Platt vs. Isotonic)...")
    calibration_results = run_probability_calibration(
        xgb_weighted_A,
        data["Policy_A"]["X_train"], data["Policy_A"]["y_train"],
        data["Policy_A"]["X_val"], data["Policy_A"]["y_val"],
        data["Policy_A"]["X_test"], data["Policy_A"]["y_test"]
    )
    
    with open(output_dir / "stage5_calibration_results.json", "w") as f:
        # Export serializable dict
        serializable_calib = {
            k: {"validation": v["validation"], "test": v["test"]}
            for k, v in calibration_results.items()
        }
        json.dump(serializable_calib, f, indent=2)
        
    print(f"  -> Calibration Comparison on Held-Out Test:")
    for k, v in calibration_results.items():
        print(f"     • {k:<15} -> Multiclass Brier: {v['test']['brier_multiclass']:.4f} | Log Loss: {v['test']['log_loss']:.4f} | Severe Brier: {v['test']['brier_severe_class4']:.4f}")
        
    # -------------------------------------------------------------------------
    # 5. Continuous Score Regression vs. Direct Classification (WP7)
    # -------------------------------------------------------------------------
    print("\n[Step 6/7] Training Direct Continuous PHQ-9 Regressors and Evaluating Discretization...")
    reg_results = run_continuous_score_regression(
        data["Policy_A"]["X_train"], data["Policy_A"]["y_score_train"],
        data["Policy_A"]["X_val"], data["Policy_A"]["y_score_val"], data["Policy_A"]["y_val"],
        data["Policy_A"]["X_test"], data["Policy_A"]["y_score_test"], data["Policy_A"]["y_test"]
    )
    
    with open(output_dir / "stage5_regression_discretization.json", "w") as f:
        # Strip numpy objects for JSON export
        clean_reg = {}
        for r_name, r_dict in reg_results.items():
            clean_reg[r_name] = {
                "fit_time_sec": r_dict["fit_time_sec"],
                "continuous_metrics": r_dict["continuous_metrics"],
                "discretized_test_summary": {
                    "macro_f1": round(r_dict["discretized_test_metrics"]["macro_f1"], 4),
                    "balanced_accuracy": round(r_dict["discretized_test_metrics"]["balanced_accuracy"], 4),
                    "accuracy": round(r_dict["discretized_test_metrics"]["accuracy"], 4),
                    "qwk": round(r_dict["discretized_test_metrics"]["quadratic_weighted_kappa"], 4),
                    "ordinal_mae": round(r_dict["discretized_test_metrics"]["ordinal_class_mae"], 4),
                    "severe_recall": round(r_dict["discretized_test_metrics"]["per_class_recall"]["4"], 4),
                    "extreme_errors_test": r_dict["extreme_errors_test"]
                }
            }
        json.dump(clean_reg, f, indent=2)
        
    print("  -> Continuous PHQ-9 Regression Results:")
    for r_name, r_dict in reg_results.items():
        cm = r_dict["continuous_metrics"]
        dm = r_dict["discretized_test_metrics"]
        print(f"     • {r_name:<30} -> Score MAE: {cm['test_score_mae']:.4f} pts | RMSE: {cm['test_score_rmse']:.4f} | Discretized Macro-F1: {dm['macro_f1']:.4f} | QWK: {dm['quadratic_weighted_kappa']:.4f}")
        
    # -------------------------------------------------------------------------
    # 6. TreeSHAP Explainability (WP6)
    # -------------------------------------------------------------------------
    print("\n[Step 7/7] Computing TreeSHAP Attributions on XGBoost Policy A...")
    shap_results = compute_stage5_treeshap(
        xgb_weighted_A,
        data["Policy_A"]["X_test"],
        data["feature_cols_A"],
        data["Policy_A"]["y_test"],
        output_dir=output_dir,
        sample_size=400
    )
    print(f"  -> TreeSHAP completed. Top 5 Global Features:")
    for i, tf in enumerate(shap_results["top_features"][:5]):
        print(f"     {i+1}. {tf['feature']:<30} [{tf['domain']}] -> {tf['mean_abs_shap_all_classes']:.4f}")
        
    # -------------------------------------------------------------------------
    # 7. Generate Standalone Interactive Research Dashboard (WP9)
    # -------------------------------------------------------------------------
    print("\nGenerating Standalone Interactive Clinical Research Dashboard...")
    dashboard_payload = {
        "benchmarks": benchmarks_list,
        "policy_comparison": {
            "policy_a_xgboost": pack_benchmark_row(res_xgb_weighted_A, "XGBoost Policy A", "Policy A"),
            "policy_b_xgboost_full": pack_benchmark_row(res_xgb_weighted_B, "XGBoost Policy B (All N=1575)", "Policy B"),
            "policy_b_xgboost_trans": {
                "model_name": "XGBoost Policy B (Transitions N=973)",
                "policy": "Policy B (Transitions)",
                "test_macro_f1": round(trans_metrics_xgb_B["macro_f1"], 4),
                "test_bal_acc": round(trans_metrics_xgb_B["balanced_accuracy"], 4),
                "test_accuracy": round(trans_metrics_xgb_B["accuracy"], 4),
                "test_qwk": round(trans_metrics_xgb_B["quadratic_weighted_kappa"], 4),
                "severe_recall": round(trans_metrics_xgb_B["per_class_recall"]["4"], 4),
                "severe_precision": round(trans_metrics_xgb_B["per_class_precision"]["4"], 4)
            },
            "persistence_baseline_trans": persist_ref
        },
        "thresholds": threshold_full_record,
        "calibration": serializable_calib,
        "regression": clean_reg,
        "shap": shap_results
    }
    
    dash_path = generate_interactive_dashboard(dashboard_payload, output_dir)
    print(f"\nStage 5 Pipeline Completed Successfully! Dashboard URL: file:///{dash_path.as_posix()}")
    return dashboard_payload


if __name__ == "__main__":
    run_stage5_full_pipeline()
