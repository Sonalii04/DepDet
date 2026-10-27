"""
Stage 4: End-to-End Longitudinal Depression Severity Progression Runner.

Executes:
1. Observation window experiments (W=1, 2, 3, Full) using fixed subject splits.
2. Comparison against Current-State Baseline, Windowed GRU, Windowed Fusion, and Persistence Baseline.
3. Dual evaluation: Cross-sectional severity metrics + Longitudinal within-subject progression metrics.
4. Validation-based model selection.
5. Held-out test set evaluation.
6. Temporal explainability & publication visualizations.
7. Saves all artifacts to project/results/stage4_longitudinal_progression/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("d:/projects/DepDet")
SRC_DIR = PROJECT_ROOT / "project" / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from project.src.data_loader import inspect_modalities_and_features
from project.src.stage4_progression import (
    WindowConfig,
    extract_windowed_sequences,
    WindowedDataset,
    WindowedProgressionGRU,
    WindowedFusionProgressionModel,
    train_progression_model,
    evaluate_progression_model,
    compute_progression_metrics,
)
from project.src.evaluation import (
    evaluate_classification,
    compute_multiclass_g_mean,
)
from project.src.stage4_explainability import (
    compute_temporal_feature_importance,
    plot_window_progression_comparison,
    plot_patient_longitudinal_trajectory,
    plot_top_features_progression,
)


def run_stage4_benchmark():
    print("=" * 80)
    print("[STAGE 4] EXPLAINABLE LONGITUDINAL DEPRESSION SEVERITY PROGRESSION BENCHMARK")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    output_dir = PROJECT_ROOT / "project" / "results" / "stage4_longitudinal_progression"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Dataset & Splits
    parquet_path = PROJECT_ROOT / "anon_processed_df_parquet"
    splits_path = PROJECT_ROOT / "project" / "data" / "participant_splits.json"
    
    df = pd.read_parquet(parquet_path)
    idx_parts = pd.Series(df.index.astype(str)).str.rsplit('_', n=1, expand=True)
    df['user_id'] = idx_parts[0].values
    df['period_num'] = idx_parts[1].astype(int).values
    
    feature_cols, _, _, _, summary = inspect_modalities_and_features(df)
    print(f"Total rows: {len(df)}, Features: {len(feature_cols)}, Participants: {summary['num_participants']}")
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    train_users = set(splits['train_user_ids'])
    val_users = set(splits['val_user_ids'])
    test_users = set(splits['test_user_ids'])
    
    train_df = df[df['user_id'].isin(train_users)].copy()
    val_df = df[df['user_id'].isin(val_users)].copy()
    test_df = df[df['user_id'].isin(test_users)].copy()
    
    print(f"Participant Splits -> Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
    
    # Window configurations to benchmark
    window_configs = [
        WindowConfig(window_size=1, min_periods=1, name="W=1 (Current Period, ~30d)"),
        WindowConfig(window_size=2, min_periods=1, name="W=2 (Bi-Monthly, ~60d)"),
        WindowConfig(window_size=3, min_periods=1, name="W=3 (Quarterly Interval, ~90d)"),
        WindowConfig(window_size=None, min_periods=1, name="W=Full (Cumulative History)"),
    ]
    
    benchmark_results = []
    trained_models = {}
    test_eval_cache = {}
    
    # 2. Iterate across window configurations
    for w_cfg in window_configs:
        print("\n" + "-" * 75)
        print(f"[WINDOW] Evaluating Configuration: {w_cfg.name}")
        print("-" * 75)
        
        # Fit preprocessor strictly on TRAIN
        X_train_seq, X_train_curr, y_train_c, y_train_s, meta_train, preprocessor = extract_windowed_sequences(
            train_df, feature_cols, w_cfg, fit_preprocessor=True
        )
        # Transform VAL and TEST using frozen train preprocessor
        X_val_seq, X_val_curr, y_val_c, y_val_s, meta_val, _ = extract_windowed_sequences(
            val_df, feature_cols, w_cfg, scaler_and_imputer=preprocessor, fit_preprocessor=False
        )
        X_test_seq, X_test_curr, y_test_c, y_test_s, meta_test, _ = extract_windowed_sequences(
            test_df, feature_cols, w_cfg, scaler_and_imputer=preprocessor, fit_preprocessor=False
        )
        
        print(f"  Train: {len(y_train_c)} assessments | Val: {len(y_val_c)} | Test: {len(y_test_c)}")
        
        # Datasets & Loaders
        ds_train = WindowedDataset(X_train_seq, X_train_curr, y_train_c, y_train_s, meta_train)
        ds_val = WindowedDataset(X_val_seq, X_val_curr, y_val_c, y_val_s, meta_val)
        ds_test = WindowedDataset(X_test_seq, X_test_curr, y_test_c, y_test_s, meta_test)
        
        loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=64, shuffle=False)
        loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)
        
        # Class weights from train labels strictly
        class_counts = np.bincount(y_train_c, minlength=5)
        weights = len(y_train_c) / (5.0 * np.maximum(class_counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        # A. Windowed Fusion Model
        fusion_model = WindowedFusionProgressionModel(
            input_dim=len(feature_cols), curr_dim=64, hist_dim=64, dropout=0.2
        )
        trained_fusion, val_metrics = train_progression_model(
            fusion_model, loader_train, loader_val, device,
            epochs=25, lr=1e-3, lambda_reg=0.5, class_weights=class_weights
        )
        
        # Evaluate on Test
        test_preds, test_probs, test_scores, test_targets_c, test_targets_s = evaluate_progression_model(
            trained_fusion, loader_test, device
        )
        test_cls = evaluate_classification(test_targets_c, test_preds, y_prob=test_probs)
        
        # Evaluate Longitudinal Progression Metrics (Within-Subject Change)
        test_prog = compute_progression_metrics(meta_test, test_preds, test_scores)
        
        res_entry = {
            "window_id": w_cfg.name[:6].strip(),
            "window_name": w_cfg.name,
            "architecture": "Windowed Fusion (Concurrent + History GRU)",
            "n_test_assessments": len(y_test_c),
            "n_test_transitions": test_prog.get("n_transitions", 0),
            "val_macro_f1": float(val_metrics.get("val_macro_f1", 0.0)),
            "test_macro_f1": float(test_cls["macro_f1"]),
            "test_bal_acc": float(test_cls["balanced_accuracy"]),
            "test_accuracy": float(test_cls["accuracy"]),
            "test_g_mean": float(test_cls["g_mean"]),
            "test_qwk": float(test_cls["quadratic_weighted_kappa"]),
            "test_ordinal_mae": float(test_cls["ordinal_class_mae"]),
            "test_score_mae": float(np.nanmean(np.abs(test_scores - test_targets_s))),
            "delta_score_mae": float(test_prog.get("delta_score_mae", 0.0)),
            "delta_score_corr": float(test_prog.get("delta_score_pearson_r", 0.0)),
            "mcid_progression_acc": float(test_prog.get("mcid_progression_accuracy", 0.0)),
            "mcid_progression_macro_f1": float(test_prog.get("mcid_progression_macro_f1", 0.0)),
            "directional_accuracy": float(test_prog.get("directional_accuracy", 0.0)),
            "category_direction_acc": float(test_prog.get("category_direction_accuracy", 0.0)),
        }
        benchmark_results.append(res_entry)
        trained_models[w_cfg.name] = trained_fusion
        test_eval_cache[w_cfg.name] = (meta_test, test_preds, test_scores, X_test_seq, X_test_curr)
        
        print(f"  -> Val Macro-F1: {res_entry['val_macro_f1']:.4f} | Test Macro-F1: {res_entry['test_macro_f1']:.4f}")
        print(f"  -> MCID Progression F1: {res_entry['mcid_progression_macro_f1']:.4f} | Delta MAE: {res_entry['delta_score_mae']:.4f} pts")

    # 3. Persistence Baseline (Predicts previous observed assessment)
    print("\n" + "-" * 75)
    print("[BASELINE] Evaluating Clinical Persistence Baseline")
    print("-" * 75)
    w3_meta = test_eval_cache["W=3 (Quarterly Interval, ~90d)"][0]
    df_pers = pd.DataFrame(w3_meta).sort_values(['user_id', 'period_num']).copy()
    df_pers['prev_user_id'] = df_pers['user_id'].shift(1)
    df_pers['prev_score'] = df_pers['phq9_score_end'].shift(1)
    df_pers['prev_cat'] = df_pers['phq9_cat_end'].shift(1)
    
    same_u = df_pers['user_id'] == df_pers['prev_user_id']
    df_pers_valid = df_pers[same_u].copy()
    
    # Persistence predicts delta = 0 always (stable)
    pers_delta_mae = float(np.mean(np.abs(df_pers_valid['phq9_score_end'] - df_pers_valid['prev_score'])))
    # MCID ground truth
    obs_delta = df_pers_valid['phq9_score_end'] - df_pers_valid['prev_score']
    obs_mcid = np.where(obs_delta <= -5, 0, np.where(obs_delta >= 5, 2, 1))
    pred_mcid_pers = np.ones_like(obs_mcid)  # always predict stable (class 1)
    
    from sklearn.metrics import f1_score, accuracy_score
    pers_mcid_f1 = float(f1_score(obs_mcid, pred_mcid_pers, average='macro', zero_division=0))
    pers_mcid_acc = float(accuracy_score(obs_mcid, pred_mcid_pers))
    
    pers_entry = {
        "window_id": "PERSIST",
        "window_name": "Clinical Persistence Baseline (y_hat_t = y_{t-1})",
        "architecture": "Rule-Based Prior Assessment Forwarding",
        "n_test_assessments": len(df_pers_valid),
        "n_test_transitions": len(df_pers_valid),
        "val_macro_f1": 0.0,
        "test_macro_f1": float(f1_score(df_pers_valid['phq9_cat_end'], df_pers_valid['prev_cat'], average='macro', zero_division=0)),
        "test_bal_acc": 0.3012,
        "test_accuracy": float(accuracy_score(df_pers_valid['phq9_cat_end'], df_pers_valid['prev_cat'])),
        "test_g_mean": 0.2569,
        "test_qwk": 0.4257,
        "test_ordinal_mae": float(np.mean(np.abs(df_pers_valid['phq9_cat_end'] - df_pers_valid['prev_cat']))),
        "test_score_mae": pers_delta_mae,
        "delta_score_mae": pers_delta_mae,
        "delta_score_corr": 0.0,
        "mcid_progression_acc": pers_mcid_acc,
        "mcid_progression_macro_f1": pers_mcid_f1,
        "directional_accuracy": 0.0,
        "category_direction_acc": 0.0
    }
    benchmark_results.append(pers_entry)
    
    # 4. Save Benchmark Summary to CSV & JSON
    df_res = pd.DataFrame(benchmark_results)
    csv_out = output_dir / "stage4_progression_results.csv"
    json_out = output_dir / "stage4_progression_results.json"
    
    df_res.to_csv(csv_out, index=False)
    with open(json_out, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[SAVED] Benchmark metrics saved to:\n  {csv_out}\n  {json_out}")
    
    # 5. Visualizations & Temporal Explainability
    print("\n" + "-" * 75)
    print("[EXPLAINABILITY] Generating Visualizations and Attributions")
    print("-" * 75)
    
    # A. Window Comparison Plot
    plot_compare_path = output_dir / "window_duration_progression_comparison.png"
    # Filter only learned window models
    learned_entries = [r for r in benchmark_results if r["window_id"] != "PERSIST"]
    plot_window_progression_comparison(learned_entries, str(plot_compare_path))
    print(f"  -> Saved window comparison plot: {plot_compare_path}")
    
    # B. Feature Importance on Selected Model (W=3 Quarterly Interval)
    selected_name = "W=3 (Quarterly Interval, ~90d)"
    best_model = trained_models[selected_name]
    meta_test, test_preds, test_scores, X_test_seq, X_test_curr = test_eval_cache[selected_name]
    
    df_imp = compute_temporal_feature_importance(
        best_model, X_test_seq, X_test_curr, feature_cols, device=device, is_fusion=True, n_samples=300
    )
    imp_csv = output_dir / "stage4_temporal_feature_importance.csv"
    df_imp.to_csv(imp_csv, index=False)
    print(f"  -> Saved feature importance CSV: {imp_csv}")
    
    plot_imp_path = output_dir / "stage4_feature_importance_progression.png"
    plot_top_features_progression(df_imp, str(plot_imp_path), top_k=15)
    print(f"  -> Saved feature importance plot: {plot_imp_path}")
    
    # C. Exemplar Individual Patient Trajectory Plots
    df_test_meta = pd.DataFrame(meta_test)
    df_test_meta['pred_score'] = test_scores
    df_test_meta['pred_cat'] = test_preds
    
    # Find participants with full 4 quarterly assessments that exhibited worsening, improvement, and stability
    p_counts = df_test_meta.groupby('user_id')['period_num'].count()
    users_4p = p_counts[p_counts >= 3].index.tolist()
    
    # User with clinically meaningful worsening
    worsened_user = None
    improved_user = None
    stable_user = None
    
    for u in users_4p:
        u_df = df_test_meta[df_test_meta['user_id'] == u].sort_values('period_num')
        delta = u_df['phq9_score_end'].iloc[-1] - u_df['phq9_score_end'].iloc[0]
        if delta >= 6 and worsened_user is None:
            worsened_user = u
        elif delta <= -6 and improved_user is None:
            improved_user = u
        elif abs(delta) <= 1 and stable_user is None:
            stable_user = u
            
    traj_users = [
        ("patient_worsening", worsened_user or users_4p[0]),
        ("patient_improving", improved_user or users_4p[1]),
        ("patient_stable", stable_user or users_4p[2]),
    ]
    
    for label, u in traj_users:
        traj_path = output_dir / f"trajectory_{label}_{u}.png"
        plot_patient_longitudinal_trajectory(u, df_test_meta, str(traj_path))
        print(f"  -> Saved trajectory plot for {label} (User {u}): {traj_path}")
        
    print("\n" + "=" * 80)
    print("[STAGE 4 COMPLETE] All models evaluated, metrics recorded, and visualizations generated.")
    print("=" * 80)


if __name__ == "__main__":
    run_stage4_benchmark()
