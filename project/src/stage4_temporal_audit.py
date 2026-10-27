"""
Stage 4 Temporal Data Audit & Feasibility Analysis.

Performs a comprehensive read-only audit of temporal resolution, observation intervals,
assessment schedules, transition pairs, and missingness in anon_processed_df_parquet.
Saves machine-readable audit to project/results/stage4_longitudinal_progression/temporal_data_audit.json.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def run_temporal_audit():
    parquet_path = "d:/projects/DepDet/anon_processed_df_parquet"
    splits_path = "d:/projects/DepDet/project/data/participant_splits.json"
    output_dir = Path("d:/projects/DepDet/project/results/stage4_longitudinal_progression")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(parquet_path)
    idx_parts = pd.Series(df.index.astype(str)).str.rsplit('_', n=1, expand=True)
    df['user_id'] = idx_parts[0].values
    df['period_num'] = idx_parts[1].astype(int).values
    
    # Load splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    train_users = set(splits['train_user_ids'])
    val_users = set(splits['val_user_ids'])
    test_users = set(splits['test_user_ids'])
    
    # Basic counts
    n_total_rows = len(df)
    n_participants = int(df['user_id'].nunique())
    
    # Check for datetime/timestamp columns
    datetime_cols = [c for c in df.columns if df[c].dtype.kind in ('M', 'm')]
    has_exact_timestamps = len(datetime_cols) > 0
    
    # Period stats
    period_counts = df['period_num'].value_counts().sort_index().to_dict()
    labeled_period_counts = df[df['phq9_cat_end'].notna()]['period_num'].value_counts().sort_index().to_dict()
    unlabeled_period_counts = df[df['phq9_cat_end'].isna()]['period_num'].value_counts().sort_index().to_dict()
    
    # Periods per user
    user_period_counts = df.groupby('user_id')['period_num'].count()
    user_periods_dist = {
        "min": int(user_period_counts.min()),
        "max": int(user_period_counts.max()),
        "mean": float(user_period_counts.mean()),
        "median": float(user_period_counts.median()),
        "std": float(user_period_counts.std())
    }
    
    # Labeled assessments per user
    df_labeled = df[df['phq9_cat_end'].notna()].copy()
    user_labeled_counts = df_labeled.groupby('user_id')['period_num'].count()
    user_labeled_dist = {
        "participants_with_at_least_1": int(len(user_labeled_counts)),
        "participants_with_1": int((user_labeled_counts == 1).sum()),
        "participants_with_2": int((user_labeled_counts == 2).sum()),
        "participants_with_3": int((user_labeled_counts == 3).sum()),
        "participants_with_4": int((user_labeled_counts == 4).sum()),
        "participants_with_at_least_2": int((user_labeled_counts >= 2).sum()),
        "participants_with_at_least_3": int((user_labeled_counts >= 3).sum())
    }
    
    # Assessment transitions (t_prev -> t_curr within same participant)
    df_labeled_sorted = df_labeled.sort_values(['user_id', 'period_num']).copy()
    df_labeled_sorted['prev_user_id'] = df_labeled_sorted['user_id'].shift(1)
    df_labeled_sorted['prev_period_num'] = df_labeled_sorted['period_num'].shift(1)
    df_labeled_sorted['prev_phq9_score_end'] = df_labeled_sorted['phq9_score_end'].shift(1)
    df_labeled_sorted['prev_phq9_cat_end'] = df_labeled_sorted['phq9_cat_end'].shift(1)
    
    same_user = df_labeled_sorted['user_id'] == df_labeled_sorted['prev_user_id']
    transitions = df_labeled_sorted[same_user].copy()
    transitions['period_gap'] = transitions['period_num'] - transitions['prev_period_num']
    transitions['delta_score'] = transitions['phq9_score_end'] - transitions['prev_phq9_score_end']
    transitions['delta_cat'] = transitions['phq9_cat_end'] - transitions['prev_phq9_cat_end']
    transitions['split'] = transitions['user_id'].map(lambda u: 'train' if u in train_users else ('val' if u in val_users else 'test'))
    
    # Gap distribution
    gap_dist = transitions['period_gap'].value_counts().sort_index().to_dict()
    
    # MCID transitions (+/- 5 points standard clinical threshold)
    transitions['mcid_direction'] = np.where(transitions['delta_score'] >= 5, 'Worsened',
                                    np.where(transitions['delta_score'] <= -5, 'Improved', 'Stable'))
    mcid_counts = transitions['mcid_direction'].value_counts().to_dict()
    mcid_props = (transitions['mcid_direction'].value_counts(normalize=True) * 100).to_dict()
    
    # Category transitions
    transitions['cat_direction'] = np.where(transitions['delta_cat'] > 0, 'Worsened',
                                   np.where(transitions['delta_cat'] < 0, 'Improved', 'Stable'))
    cat_counts = transitions['cat_direction'].value_counts().to_dict()
    cat_props = (transitions['cat_direction'].value_counts(normalize=True) * 100).to_dict()
    
    # Transitions by split
    split_transition_counts = transitions['split'].value_counts().to_dict()
    split_transition_participants = transitions.groupby('split')['user_id'].nunique().to_dict()
    
    audit_data = {
        "dataset_summary": {
            "total_rows": n_total_rows,
            "total_participants": n_participants,
            "has_exact_timestamps": has_exact_timestamps,
            "has_daily_measurements": False,
            "has_weekly_measurements": False,
            "temporal_structure": "Period-Level Aggregates (12 Monthly Periods; Quarterly PHQ-9 Assessments at Periods 3, 6, 9, 12)",
            "feasibility_case": "Case C: Only period-level aggregates are available"
        },
        "period_breakdown": {
            "total_rows_by_period": {str(k): int(v) for k, v in period_counts.items()},
            "labeled_rows_by_period": {str(k): int(v) for k, v in labeled_period_counts.items()},
            "unlabeled_rows_by_period": {str(k): int(v) for k, v in unlabeled_period_counts.items()}
        },
        "participant_coverage": {
            "periods_per_user": user_periods_dist,
            "labeled_assessments_per_user": user_labeled_dist
        },
        "longitudinal_transitions": {
            "total_transitions": int(len(transitions)),
            "transitions_by_split": {str(k): int(v) for k, v in split_transition_counts.items()},
            "transition_participants_by_split": {str(k): int(v) for k, v in split_transition_participants.items()},
            "period_gap_distribution": {str(k): int(v) for k, v in gap_dist.items()},
            "delta_score_stats": {
                "mean": float(transitions['delta_score'].mean()),
                "std": float(transitions['delta_score'].std()),
                "median": float(transitions['delta_score'].median()),
                "q25": float(transitions['delta_score'].quantile(0.25)),
                "q75": float(transitions['delta_score'].quantile(0.75))
            },
            "mcid_5pt_distribution": {
                "counts": {str(k): int(v) for k, v in mcid_counts.items()},
                "percentages": {str(k): float(v) for k, v in mcid_props.items()}
            },
            "category_change_distribution": {
                "counts": {str(k): int(v) for k, v in cat_counts.items()},
                "percentages": {str(k): float(v) for k, v in cat_props.items()}
            }
        },
        "supported_observation_windows": [
            {
                "window_id": "W1",
                "name": "Current-Period Window (W=1)",
                "duration_periods": 1,
                "approx_duration_days": 30,
                "description": "Features from assessment period t only (concurrent digital phenotype)."
            },
            {
                "window_id": "W2",
                "name": "Bi-Monthly Window (W=2)",
                "duration_periods": 2,
                "approx_duration_days": 60,
                "description": "Features from preceding period (t-1) and assessment period t."
            },
            {
                "window_id": "W3",
                "name": "Quarterly Assessment-Interval Window (W=3)",
                "duration_periods": 3,
                "approx_duration_days": 90,
                "description": "Full 3-period block [t-2, t-1, t] spanning the entire inter-assessment interval."
            },
            {
                "window_id": "W_all",
                "name": "Cumulative History Window (W=Full)",
                "duration_periods": "variable",
                "approx_duration_days": "up to 360",
                "description": "All available observation periods from study baseline up to assessment period t."
            }
        ]
    }
    
    out_file = output_dir / "temporal_data_audit.json"
    with open(out_file, 'w') as f:
        json.dump(audit_data, f, indent=2)
    print(f"Audit successfully saved to: {out_file}")
    return audit_data

if __name__ == "__main__":
    run_temporal_audit()
