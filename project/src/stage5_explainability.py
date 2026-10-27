"""
Stage 5: Explainability & TreeSHAP Integration (WP6).

Implements:
1. TreeSHAP computation on trained XGBoost models using shap.TreeExplainer.
2. Global feature importance summaries & domain-level attribution percentages.
3. Local waterfall/force explanations for exemplar test cases:
   - Severe Depression True Positive Alert
   - Severe Depression False Positive Alert
   - Severe Depression False Negative (Missed Severe Case)
4. Model-agnostic feature attributions with strict non-causal interpretation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def map_feature_to_domain(feature_name: str) -> str:
    """
    Maps a feature name into one of 4 functional clinical domains.
    """
    f_lower = feature_name.lower()
    
    if any(k in f_lower for k in ['sleep', 'bed', 'asleep', 'circadian', 'hyposomnia', 'hypersomnia', 'nap']):
        return "Sleep & Circadian"
    elif any(k in f_lower for k in ['step', 'mvpa', 'lpa', 'sedentary', 'active', 'light_activity']):
        return "Actigraphy & Physical Activity"
    elif any(k in f_lower for k in ['sex', 'birth', 'age', 'educ', 'height', 'weight', 'bmi', 'pregnant', 'money', 'household', 'insurance']):
        return "Demographics & Socioeconomic"
    else:
        return "Clinical Comorbidities & Health History"


def compute_stage5_treeshap(
    xgb_model,
    X_test: np.ndarray,
    feature_names: list[str],
    y_test: np.ndarray,
    output_dir: Path,
    sample_size: int = 400
) -> dict:
    """
    Computes exact TreeSHAP values for XGBoost on a representative test sample.
    Exports global rankings, domain breakdown, and local case explanations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Subsample test set deterministically for SHAP computation
    np.random.seed(42)
    n_samples = min(sample_size, len(X_test))
    indices = np.random.choice(len(X_test), size=n_samples, replace=False)
    
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP values shape: (N, D, 5) or list of 5 arrays (N, D) depending on XGBoost version
    if isinstance(shap_values, list):
        # Shape: list of 5 arrays, each (N, D)
        shap_severe = shap_values[4]  # Class 4 (Severe Depression)
        shap_global_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0) # (N, D)
    elif len(shap_values.shape) == 3:
        shap_severe = shap_values[:, :, 4]
        shap_global_mean = np.mean(np.abs(shap_values), axis=2) # (N, D)
    else:
        shap_severe = shap_values
        shap_global_mean = np.abs(shap_values)
        
    # Global mean absolute SHAP value per feature
    mean_abs_shap = np.mean(shap_global_mean, axis=0)
    mean_abs_severe_shap = np.mean(np.abs(shap_severe), axis=0)
    
    ranking_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_all_classes": mean_abs_shap,
        "mean_abs_shap_severe_class4": mean_abs_severe_shap,
        "domain": [map_feature_to_domain(f) for f in feature_names]
    }).sort_values("mean_abs_shap_all_classes", ascending=False).reset_index(drop=True)
    
    ranking_csv = output_dir / "stage5_shap_feature_importance.csv"
    ranking_df.to_csv(ranking_csv, index=False)
    
    # Domain-level attribution summary
    domain_summary = ranking_df.groupby("domain")["mean_abs_shap_all_classes"].agg(["sum", "count"]).reset_index()
    total_shap = domain_summary["sum"].sum()
    domain_summary["percentage_contribution"] = (domain_summary["sum"] / total_shap * 100.0).round(2)
    domain_summary = domain_summary.sort_values("percentage_contribution", ascending=False).reset_index(drop=True)
    
    domain_csv = output_dir / "stage5_domain_importance.csv"
    domain_summary.to_csv(domain_csv, index=False)
    
    # Plot 1: Top 20 Global Features Bar Chart
    plt.figure(figsize=(10, 8))
    top20 = ranking_df.head(20)
    colors = {
        "Sleep & Circadian": "#4575b4",
        "Actigraphy & Physical Activity": "#74add1",
        "Demographics & Socioeconomic": "#fdae61",
        "Clinical Comorbidities & Health History": "#d73027"
    }
    bar_colors = [colors.get(d, "#999999") for d in top20["domain"]]
    
    plt.barh(top20["feature"][::-1], top20["mean_abs_shap_all_classes"][::-1], color=bar_colors[::-1], edgecolor="black", linewidth=0.5)
    plt.xlabel("Mean Absolute SHAP Value (Overall Severity)", fontsize=11)
    plt.title("Stage 5: Top 20 Predictive Features (TreeSHAP - XGBoost Policy A)", fontsize=12, fontweight="bold")
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=col) for col in colors.values()]
    plt.legend(handles, colors.keys(), loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "stage5_shap_top20_global.png", dpi=200)
    plt.close()
    
    # Plot 2: Domain Importance Breakdown
    plt.figure(figsize=(8, 5))
    sns.barplot(data=domain_summary, x="percentage_contribution", y="domain", palette="Blues_r")
    plt.xlabel("Predictive Information Share (% of Total SHAP Importance)", fontsize=11)
    plt.ylabel("")
    plt.title("Predictive Contribution by Clinical Domain (TreeSHAP)", fontsize=12, fontweight="bold")
    for i, row in domain_summary.iterrows():
        plt.text(row["percentage_contribution"] + 0.5, i, f"{row['percentage_contribution']:.1f}%", va="center", fontweight="bold")
    plt.xlim(0, max(domain_summary["percentage_contribution"]) * 1.2)
    plt.tight_layout()
    plt.savefig(plots_dir / "stage5_domain_importance_breakdown.png", dpi=200)
    plt.close()
    
    # Identify local exemplar cases from the sample
    preds_sample = np.argmax(xgb_model.predict_proba(X_sample), axis=1)
    
    tp_idx = np.where((y_sample == 4) & (preds_sample == 4))[0]
    fp_idx = np.where((y_sample != 4) & (preds_sample == 4))[0]
    fn_idx = np.where((y_sample == 4) & (preds_sample != 4))[0]
    
    def extract_case_attributions(idx_list, case_name):
        if len(idx_list) == 0:
            return None
        c_idx = int(idx_list[0])
        case_shap = shap_severe[c_idx]
        sorted_feat_idx = np.argsort(np.abs(case_shap))[::-1][:10]
        top_local = [
            {
                "feature": feature_names[fi],
                "feature_value_scaled": round(float(X_sample[c_idx, fi]), 4),
                "shap_impact_on_severe": round(float(case_shap[fi]), 4),
                "domain": map_feature_to_domain(feature_names[fi])
            }
            for fi in sorted_feat_idx
        ]
        return {
            "case_type": case_name,
            "true_severity_tier": int(y_sample[c_idx]),
            "predicted_severity_tier": int(preds_sample[c_idx]),
            "top_10_driving_features": top_local
        }
        
    local_cases = {
        "true_positive_severe_alert": extract_case_attributions(tp_idx, "True Positive Severe Alert"),
        "false_positive_severe_alert": extract_case_attributions(fp_idx, "False Positive Severe Alert"),
        "false_negative_missed_severe": extract_case_attributions(fn_idx, "False Negative Missed Severe")
    }
    
    with open(output_dir / "stage5_local_explanations.json", "w") as f:
        json.dump(local_cases, f, indent=2)
        
    return {
        "top_features": ranking_df.head(15).to_dict(orient="records"),
        "domain_summary": domain_summary.to_dict(orient="records"),
        "local_cases": local_cases
    }
