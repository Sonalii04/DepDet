import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import os
from data_loader import load_raw_dataset, inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from baseline import prepare_dataset_arrays, BaselineMLP

def compute_permutation_importance(model, X_test, y_test, feature_names, save_dir="d:/projects/DepDet/project/results/shap"):
    """
    Computes permutation feature importance on test set.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(X_test, dtype=torch.float32))
        baseline_preds = torch.argmax(logits, dim=1).numpy()
    baseline_acc = (baseline_preds == y_test).mean()
    
    importances = []
    np.random.seed(42)
    
    # Subsample 50 features for speed
    for col_idx in range(X_test.shape[1]):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, col_idx])
        with torch.no_grad():
            perm_logits, _, _ = model(torch.tensor(X_perm, dtype=torch.float32))
            perm_preds = torch.argmax(perm_logits, dim=1).numpy()
        perm_acc = (perm_preds == y_test).mean()
        drop = baseline_acc - perm_acc
        importances.append(drop)
        
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_drop": importances
    }).sort_values(by="importance_drop", ascending=False)
    
    top_20 = imp_df.head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance_drop", y="feature", data=top_20, palette="viridis")
    plt.title("Permutation Feature Importance (Accuracy Drop on Test)", fontsize=14, fontweight='bold')
    plt.xlabel("Accuracy Decrease on Permutation", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "permutation_importance_top20.png"), dpi=300)
    plt.close()
    
    imp_df.to_csv(os.path.join(save_dir, "permutation_importance.csv"), index=False)
    print("Saved permutation feature importance plots and CSV.")
    return imp_df

def compute_shap_explanations(model, X_train, X_test, feature_names, save_dir="d:/projects/DepDet/project/results/shap"):
    """
    Computes global and per-patient SHAP feature importances using fast sampling.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    def model_predict(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            logits, _, _ = model(x_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs
        
    np.random.seed(42)
    bg_idx = np.random.choice(X_train.shape[0], size=10, replace=False)
    test_idx = np.random.choice(X_test.shape[0], size=15, replace=False)
    
    background = X_train[bg_idx]
    test_sample = X_test[test_idx]
    
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_sample, nsamples=40)
    
    # 1. Global Feature Importance Bar Plot
    plt.figure(figsize=(10, 8))
    sv_severe = shap_values[4] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(
        sv_severe,
        test_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Global Feature Importance (Severe Depression Class)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_global_bar.png"), dpi=300)
    plt.close()
    
    # 2. Global Feature Summary Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_severe,
        test_sample,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Global Feature Impact (Severe Depression Class)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_global_beeswarm.png"), dpi=300)
    plt.close()
    
    # 3. Local Per-Patient Feature Attribution
    for p_idx in [0, 1]:
        plt.figure(figsize=(10, 6))
        patient_shap = np.array(sv_severe[p_idx])
        if patient_shap.ndim > 1:
            patient_shap = patient_shap.mean(axis=-1)
        patient_shap = patient_shap.flatten()[:len(feature_names)]
        
        patient_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": patient_shap,
            "abs_shap": np.abs(patient_shap)
        }).sort_values(by="abs_shap", ascending=False).head(10)
        
        sns.barplot(x="shap_value", y="feature", data=patient_df, palette="coolwarm")
        plt.title(f"Patient #{p_idx+1} Local Feature Attribution (Severe Class)", fontsize=14, fontweight='bold')
        plt.xlabel("SHAP Value (Contribution to Depression Risk)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"patient_{p_idx+1}_local_shap.png"), dpi=300)
        plt.close()
        
    print("Saved SHAP global and local explanation plots.")

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    
    prep = DataPreprocessor(feature_cols=f_cols)
    X_train, y_train_c, _, _, _ = prepare_dataset_arrays(train_df, prep, fit_preprocessor=True)
    X_test, y_test_c, _, _, _ = prepare_dataset_arrays(test_df, prep, fit_preprocessor=False)
    
    model = BaselineMLP(input_dim=X_train.shape[1], hidden_dim=128, num_classes=5)
    ckpt = torch.load("d:/projects/DepDet/project/checkpoints/baseline_best.pt")
    model.load_state_dict(ckpt)
    
    compute_permutation_importance(model, X_test, y_test_c, f_cols)
    compute_shap_explanations(model, X_train, X_test, f_cols)
