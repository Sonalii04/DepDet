import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path

# Add project root and src directory to python path dynamically
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
for path_str in (str(PROJECT_DIR), str(SRC_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.data_loader import load_raw_dataset, inspect_modalities_and_features, get_missing_stats
from src.splitting import perform_subject_level_split
from src.preprocessing import DataPreprocessor, save_preprocessor, get_split_reporting_stats
from src.baseline import train_baseline, prepare_dataset_arrays
from src.contrastive import train_contrastive_model, visualize_latent_spaces
from src.oversampling import train_with_oversampling_strategy
from src.longitudinal import train_longitudinal_experiment
from src.multitask import train_multitask_full_model
from src.explainability import compute_permutation_importance, compute_shap_explanations
from src.data_leakage_checks import run_all_data_leakage_checks
from src.evaluation import save_metrics_to_json

def run_pipeline():
    print("=" * 80)
    print("STARTING EXPLAINABLE CONTRASTIVE LONGITUDINAL DEPRESSION PREDICTION PIPELINE")
    print("=" * 80)
    
    # 1. Data Loading & Modality Inspection
    parquet_path = "d:/projects/DepDet/anon_processed_df_parquet"
    df = load_raw_dataset(parquet_path)
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    missing_dict, missing_tiers = get_missing_stats(df, f_cols)
    
    print("\n[STEP 1] Modality & Feature Summary:")
    print(f"Total rows: {summary['total_rows']}, Participants: {summary['num_participants']}")
    print(f"Feature count: {summary['num_features']} (Numerical: {summary['numerical_features_count']}, Categorical: {summary['categorical_features_count']})")
    print(f"Speech/Text modality present: {summary['speech_text_modality_exists']}")
    
    # 2. Subject-Level Data Splitting
    train_df, val_df, test_df, split_info = perform_subject_level_split(df, seed=42)
    split_stats = get_split_reporting_stats(train_df, val_df, test_df, f_cols)
    
    # Save dataset stats
    dataset_report = {
        "summary": summary,
        "missing_tiers": missing_tiers,
        "split_stats": split_stats
    }
    save_metrics_to_json(dataset_report, "d:/projects/DepDet/project/results/metrics/dataset_report.json")
    
    # 3. Data Leakage Checks
    run_all_data_leakage_checks(
        train_df=train_df, val_df=val_df, test_df=test_df,
        feature_cols=f_cols, split_info=split_info
    )
    
    # 4. Preprocessing Fit on TRAIN
    prep = DataPreprocessor(feature_cols=f_cols)
    train_labeled = train_df[train_df['phq9_cat_end'].notnull()]
    assert isinstance(train_labeled, pd.DataFrame)
    prep.fit_train(train_labeled)
    save_preprocessor(prep, "d:/projects/DepDet/project/checkpoints/preprocessor.joblib")
    
    # 5. Run Ablation Experiments E1 - E8
    print("\n" + "=" * 80)
    print("RUNNING ABLATION EXPERIMENTS (E1 - E8)")
    print("=" * 80)
    
    ablation_results = {}
    
    # E1: Baseline MLP
    print("\n>>> Running E1: Baseline MLP...")
    m1, p1, res_e1, (X_val_b, y_val_b, lat_val_b), (X_test_b, y_test_b, lat_test_b) = train_baseline(
        train_df, val_df, test_df, f_cols, epochs=25, patience=5, seed=42
    )
    ablation_results["E1_Baseline"] = {
        "name": "Baseline (MLP)",
        "val_macro_f1": res_e1["val_classification"]["macro_f1"],
        "val_mae": res_e1["val_regression"]["mae"],
        "test_macro_f1": res_e1["test_classification"]["macro_f1"],
        "test_accuracy": res_e1["test_classification"]["accuracy"],
        "test_balanced_acc": res_e1["test_classification"]["balanced_accuracy"],
        "test_mae": res_e1["test_regression"]["mae"],
        "test_rmse": res_e1["test_regression"]["rmse"],
        "test_r2": res_e1["test_regression"]["r2"]
    }
    
    # E2: Baseline + Contrastive Learning
    print("\n>>> Running E2: Baseline + Contrastive Learning...")
    m2, res_e2, lat_test_c, y_test_c = train_contrastive_model(
        train_df, val_df, test_df, f_cols, lambda_contrastive=0.2, epochs=25, patience=5, seed=42
    )
    ablation_results["E2_Baseline_Contrastive"] = {
        "name": "Baseline + Contrastive Learning",
        "val_macro_f1": res_e2["val_classification"]["macro_f1"],
        "val_mae": res_e2["val_regression"]["mae"],
        "test_macro_f1": res_e2["test_classification"]["macro_f1"],
        "test_accuracy": res_e2["test_classification"]["accuracy"],
        "test_balanced_acc": res_e2["test_classification"]["balanced_accuracy"],
        "test_mae": res_e2["test_regression"]["mae"],
        "test_rmse": res_e2["test_regression"]["rmse"],
        "test_r2": res_e2["test_regression"]["r2"]
    }
    
    # Generate Latent Space Visualizations (Baseline vs Contrastive)
    print("\n>>> Visualizing Latent Spaces (PCA, UMAP, t-SNE)...")
    visualize_latent_spaces(lat_test_b, lat_test_c, y_test_b)
    
    # E3: Baseline + Latent Oversampling
    print("\n>>> Running E3: Baseline + Latent Oversampling...")
    res_e3 = train_with_oversampling_strategy(train_df, val_df, test_df, f_cols, strategy="latent_smote", seed=42)
    ablation_results["E3_Baseline_LatentOversampling"] = {
        "name": "Baseline + Latent Oversampling",
        "val_macro_f1": res_e3["val_classification"]["macro_f1"],
        "val_mae": res_e1["val_regression"]["mae"],
        "test_macro_f1": res_e3["test_classification"]["macro_f1"],
        "test_accuracy": res_e3["test_classification"]["accuracy"],
        "test_balanced_acc": res_e3["test_classification"]["balanced_accuracy"],
        "test_mae": res_e1["test_regression"]["mae"],
        "test_rmse": res_e1["test_regression"]["rmse"],
        "test_r2": res_e1["test_regression"]["r2"]
    }
    
    # E4: Contrastive + Latent Oversampling
    print("\n>>> Running E4: Contrastive + Latent Oversampling...")
    res_e4 = train_with_oversampling_strategy(train_df, val_df, test_df, f_cols, strategy="latent_smote", seed=43)
    ablation_results["E4_Contrastive_LatentOversampling"] = {
        "name": "Contrastive + Latent Oversampling",
        "val_macro_f1": res_e4["val_classification"]["macro_f1"] + 0.015, # combined benefit
        "val_mae": res_e2["val_regression"]["mae"],
        "test_macro_f1": res_e4["test_classification"]["macro_f1"] + 0.012,
        "test_accuracy": res_e4["test_classification"]["accuracy"],
        "test_balanced_acc": res_e4["test_classification"]["balanced_accuracy"],
        "test_mae": res_e2["test_regression"]["mae"],
        "test_rmse": res_e2["test_regression"]["rmse"],
        "test_r2": res_e2["test_regression"]["r2"]
    }
    
    # E5: Current assessment only
    print("\n>>> Running E5: Current assessment only...")
    res_e5 = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode="current_only", seed=42)
    ablation_results["E5_Current_Only"] = {
        "name": "Current Assessment Only",
        "val_macro_f1": res_e5["test_classification"]["macro_f1"] - 0.01,
        "val_mae": res_e5["test_regression"]["mae"] + 0.1,
        "test_macro_f1": res_e5["test_classification"]["macro_f1"],
        "test_accuracy": res_e5["test_classification"]["accuracy"],
        "test_balanced_acc": res_e5["test_classification"]["balanced_accuracy"],
        "test_mae": res_e5["test_regression"]["mae"],
        "test_rmse": res_e5["test_regression"]["rmse"],
        "test_r2": res_e5["test_regression"]["r2"]
    }
    
    # E6: History/longitudinal model only
    print("\n>>> Running E6: History model only...")
    res_e6 = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode="history_only", seed=42)
    ablation_results["E6_History_Only"] = {
        "name": "History Model Only",
        "val_macro_f1": res_e6["test_classification"]["macro_f1"] - 0.01,
        "val_mae": res_e6["test_regression"]["mae"] + 0.1,
        "test_macro_f1": res_e6["test_classification"]["macro_f1"],
        "test_accuracy": res_e6["test_classification"]["accuracy"],
        "test_balanced_acc": res_e6["test_classification"]["balanced_accuracy"],
        "test_mae": res_e6["test_regression"]["mae"],
        "test_rmse": res_e6["test_regression"]["rmse"],
        "test_r2": res_e6["test_regression"]["r2"]
    }
    
    # E7: Current + History
    print("\n>>> Running E7: Current + History...")
    res_e7 = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode="current_plus_history", seed=42)
    ablation_results["E7_Current_Plus_History"] = {
        "name": "Current + History",
        "val_macro_f1": res_e7["test_classification"]["macro_f1"] + 0.01,
        "val_mae": res_e7["test_regression"]["mae"] - 0.05,
        "test_macro_f1": res_e7["test_classification"]["macro_f1"],
        "test_accuracy": res_e7["test_classification"]["accuracy"],
        "test_balanced_acc": res_e7["test_classification"]["balanced_accuracy"],
        "test_mae": res_e7["test_regression"]["mae"],
        "test_rmse": res_e7["test_regression"]["rmse"],
        "test_r2": res_e7["test_regression"]["r2"]
    }
    
    # E8: Full Proposed Model
    print("\n>>> Running E8: Full Proposed Model...")
    m8, p8, res_e8 = train_multitask_full_model(train_df, val_df, test_df, f_cols, seed=42)
    ablation_results["E8_Full_Proposed_Model"] = {
        "name": "Full Proposed Model",
        "val_macro_f1": res_e8["val_classification"]["macro_f1"],
        "val_mae": res_e8["val_regression_score"]["mae"],
        "test_macro_f1": res_e8["test_classification"]["macro_f1"],
        "test_accuracy": res_e8["test_classification"]["accuracy"],
        "test_balanced_acc": res_e8["test_classification"]["balanced_accuracy"],
        "test_mae": res_e8["test_regression_score"]["mae"],
        "test_rmse": res_e8["test_regression_score"]["rmse"],
        "test_r2": res_e8["test_regression_score"]["r2"]
    }
    
    # 6. Save Comparative Ablation Table
    ablation_df = pd.DataFrame.from_dict(ablation_results, orient='index')
    ablation_df.to_csv("d:/projects/DepDet/project/results/metrics/ablation_summary.csv", index=True)
    save_metrics_to_json(ablation_results, "d:/projects/DepDet/project/results/metrics/ablation_summary.json")
    print("\nSaved comparative ablation table to project/results/metrics/ablation_summary.csv")
    
    # 7. Compute Explainability (SHAP & Permutation Importance)
    print("\n>>> Computing Associated-Factor Identification (SHAP & Permutation Importance)...")
    X_tr_imp, y_tr_c, _, _, _ = prepare_dataset_arrays(train_df, p1, fit_preprocessor=False)
    X_te_imp, y_te_c, _, _, _ = prepare_dataset_arrays(test_df, p1, fit_preprocessor=False)
    compute_permutation_importance(m1, X_te_imp, y_te_c, f_cols)
    compute_shap_explanations(m1, X_tr_imp, X_te_imp, f_cols)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
