"""
Stage 4 Temporal Explainability & Progression Visualization Module.

Generates:
1. Global feature attributions by observation window and feature modality.
2. Temporal delta explanations explaining why predicted severity changed between consecutive assessments.
3. Individual patient longitudinal progression trajectory plots with feature attribution decomposition.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Set clean matplotlib styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


FEATURE_GROUPS = {
    "Actigraphy": [
        'steps_awake_mean', 'steps_mvpa_sum_recent', 'steps_lpa_sum_recent',
        'steps__active_day_count_', 'steps__sedentary_day_count_',
        'steps_rolling_6_median_recent', 'steps_rolling_6_max_recent'
    ],
    "Sleep & Circadian": [
        'sleep_asleep_weekday_mean', 'sleep_asleep_weekend_mean',
        'sleep_in_bed_weekday_mean', 'sleep_in_bed_weekend_mean',
        'sleep_ratio_asleep_in_bed_weekday_mean', 'sleep_ratio_asleep_in_bed_weekend_mean',
        'sleep_main_start_hour_adj_median', 'sleep_asleep_iqr',
        'sleep__hypersomnia_count_', 'sleep__hyposomnia_count_'
    ],
    "Demographics": [
        'sex', 'birthyear', 'educ', 'height', 'weight', 'bmi', 'money', 'money_assistance'
    ],
    "Clinical & Lifestyle": [
        'comorbid_migraines', 'num_migraine_days', 'comorbid_neuropathic',
        'comorbid_arthritis', 'med_start', 'life_stress'
    ]
}


def compute_temporal_feature_importance(
    model,
    X_seq: np.ndarray,
    X_curr: np.ndarray,
    feature_names: List[str],
    device,
    is_fusion: bool = True,
    n_samples: int = 300
) -> pd.DataFrame:
    """
    Computes permutation feature importance on the concurrent and historical representations.
    Evaluates impact on predicted continuous PHQ-9 score.
    """
    import torch
    model.eval()
    
    n_eval = min(len(X_seq), n_samples)
    X_seq_sample = torch.tensor(X_seq[:n_eval], dtype=torch.float32).to(device)
    X_curr_sample = torch.tensor(X_curr[:n_eval], dtype=torch.float32).to(device)
    seq_lens = torch.tensor([X_seq.shape[1]] * n_eval, dtype=torch.long).to(device)
    
    with torch.no_grad():
        if is_fusion:
            _, base_scores, _ = model(X_seq_sample, X_curr_sample, seq_lens)
        else:
            _, base_scores, _ = model(X_seq_sample, seq_lens)
    base_scores = base_scores.cpu().numpy()
    
    importances = []
    
    # Evaluate top features
    for feat_idx, feat_name in enumerate(feature_names):
        # Permute feature in current assessment
        X_curr_perm = X_curr_sample.clone()
        perm_idx = torch.randperm(n_eval)
        X_curr_perm[:, feat_idx] = X_curr_perm[perm_idx, feat_idx]
        
        with torch.no_grad():
            if is_fusion:
                _, perm_scores, _ = model(X_seq_sample, X_curr_perm, seq_lens)
            else:
                X_seq_perm = X_seq_sample.clone()
                X_seq_perm[:, -1, feat_idx] = X_seq_perm[perm_idx, -1, feat_idx]
                _, perm_scores, _ = model(X_seq_perm, seq_lens)
        perm_scores = perm_scores.cpu().numpy()
        
        # Mean absolute difference in predicted score
        mae_shift = np.mean(np.abs(perm_scores - base_scores))
        
        # Determine modality
        modality = "Other"
        for mod, cols in FEATURE_GROUPS.items():
            if any(c in feat_name for c in cols):
                modality = mod
                break
                
        importances.append({
            "feature": feat_name,
            "importance_score_shift": float(mae_shift),
            "modality": modality
        })
        
    df_imp = pd.DataFrame(importances).sort_values("importance_score_shift", ascending=False)
    return df_imp


def plot_window_progression_comparison(
    window_results: List[Dict[str, Any]],
    output_path: str
):
    """
    Plots performance comparison across observation window durations W=1, 2, 3, Full.
    """
    names = [r["window_name"] for r in window_results]
    macro_f1s = [r["test_macro_f1"] for r in window_results]
    bal_accs = [r["test_bal_acc"] for r in window_results]
    mcid_f1s = [r["mcid_progression_macro_f1"] for r in window_results]
    delta_maes = [r["delta_score_mae"] for r in window_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Classification Metrics
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, macro_f1s, width, label='Test Macro-F1', color='#4361ee', alpha=0.9)
    axes[0].bar(x + width/2, bal_accs, width, label='Test Balanced Accuracy', color='#3a0ca3', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha='right')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Severity Category Classification by Window')
    axes[0].set_ylim(0.2, 0.45)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 2. Progression Directional F1 (MCID)
    axes[1].bar(names, mcid_f1s, color='#06d6a0', width=0.5, alpha=0.9)
    axes[1].set_ylabel('MCID Progression Macro-F1')
    axes[1].set_title('Clinical Progression Direction (Improved/Stable/Worsened)')
    axes[1].set_ylim(0.3, 0.7)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 3. Delta Score Prediction MAE
    axes[2].plot(names, delta_maes, marker='o', linewidth=2.5, markersize=8, color='#ef476f')
    axes[2].set_ylabel('Score Change MAE (Points)')
    axes[2].set_title(r'Prediction Error on $\Delta$ PHQ-9 Score')
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_patient_longitudinal_trajectory(
    user_id: str,
    patient_df: pd.DataFrame,
    output_path: str
):
    """
    Plots individual patient longitudinal depression severity trajectory over time:
    Contrasts actual observed PHQ-9 score against model-predicted score,
    and highlights the timing of clinically meaningful changes (MCID).
    """
    df_u = patient_df[patient_df['user_id'] == user_id].sort_values('period_num')
    
    periods = df_u['period_num'].values
    actual_scores = df_u['phq9_score_end'].values
    pred_scores = df_u['pred_score'].values
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Actual clinical assessments (scatter points connected by line)
    ax.plot(periods, actual_scores, marker='s', markersize=9, linewidth=2, color='#2b2d42',
            label='Observed PHQ-9 Assessment (Clinical Survey)', zorder=4)
            
    # Model predictions
    ax.plot(periods, pred_scores, marker='o', markersize=8, linewidth=2, linestyle='--', color='#4361ee',
            label='Predicted PHQ-9 Score (Model Output)', zorder=3)
            
    # Add clinical severity bands
    ax.axhspan(0, 4, color='#2a9d8f', alpha=0.10, label='Minimal (0–4)')
    ax.axhspan(5, 9, color='#e9c46a', alpha=0.10, label='Mild (5–9)')
    ax.axhspan(10, 14, color='#f4a261', alpha=0.10, label='Moderate (10–14)')
    ax.axhspan(15, 19, color='#e76f51', alpha=0.10, label='Mod. Severe (15–19)')
    ax.axhspan(20, 27, color='#d62828', alpha=0.10, label='Severe (20–27)')
    
    ax.set_xlabel('Observation Period (Months into Study)', fontsize=11)
    ax.set_ylabel('PHQ-9 Continuous Score (0–27)', fontsize=11)
    ax.set_title(f'Participant {user_id}: Observed vs. Predicted Longitudinal Severity Progression', fontsize=12, pad=10)
    ax.set_xticks(periods)
    ax.set_xticklabels([f'P{p}\n(M{p})' for p in periods])
    ax.set_ylim(-0.5, 27.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_features_progression(
    df_imp: pd.DataFrame,
    output_path: str,
    top_k: int = 15
):
    """Plots top driving features for longitudinal severity progression."""
    top_df = df_imp.head(top_k).sort_values("importance_score_shift", ascending=True)
    
    color_map = {
        "Actigraphy": "#4361ee",
        "Sleep & Circadian": "#7209b7",
        "Demographics": "#f72585",
        "Clinical & Lifestyle": "#4cc9f0",
        "Other": "#adb5bd"
    }
    colors = [color_map.get(m, "#adb5bd") for m in top_df["modality"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_df["feature"], top_df["importance_score_shift"], color=colors, alpha=0.85)
    
    ax.set_xlabel('Mean Absolute Score Impact on PHQ-9 (Points)', fontsize=11)
    ax.set_title('Top Features Driving Longitudinal Severity Estimation', fontsize=12, pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Legend for modalities
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=m) for m, c in color_map.items() if m in top_df["modality"].values]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
