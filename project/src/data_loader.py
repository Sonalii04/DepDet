import pandas as pd
import numpy as np
import os
import json

def load_raw_dataset(parquet_path: str) -> pd.DataFrame:
    """
    Loads raw dataset from Parquet file, inspects columns and index,
    and returns a cleaned DataFrame with parsed user_id and period_num.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    
    # Parse index into user_id and period_num
    idx_series = pd.Series(df.index.astype(str))
    parts = idx_series.str.rsplit('_', n=1, expand=True)
    df['user_id'] = parts[0].values
    df['period_num'] = parts[1].astype(int).values
    
    return df

def inspect_modalities_and_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], dict]:
    """
    Inspects modalities, checks speech/text existence, and categorizes features.
    """
    all_cols = df.columns.tolist()
    
    # Target columns
    target_cols = ['phq9_score_start', 'phq9_score_end', 'phq9_cat_start', 'phq9_cat_end']
    
    # Identifier columns
    id_cols = ['user_id', 'period_num']
    
    # Check for speech / text / audio / transcript columns
    speech_keywords = ['speech', 'text', 'transcript', 'audio', 'voice', 'interview', 'word', 'nlp', 'audio_']
    speech_text_cols = [c for c in all_cols if any(k in c.lower() for k in speech_keywords)]
    
    speech_text_exists = len(speech_text_cols) > 0
    
    # Feature columns (exclude target and id columns)
    exclude_cols = set(target_cols + id_cols)
    feature_cols = [c for c in all_cols if c not in exclude_cols]
    
    # Separate numerical and categorical features
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object' or df[c].dtype.name == 'category' or df[c].dtype == 'string']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    # Calculate derived target: delta_phq9
    df['delta_phq9'] = df['phq9_score_end'] - df['phq9_score_start']
    
    inspection_summary = {
        "total_rows": len(df),
        "total_columns": len(all_cols),
        "num_participants": int(df['user_id'].nunique()),  # pyright: ignore[reportArgumentType]
        "min_period": int(df['period_num'].min()),  # pyright: ignore[reportArgumentType]
        "max_period": int(df['period_num'].max()),  # pyright: ignore[reportArgumentType]
        "speech_text_modality_exists": speech_text_exists,
        "speech_text_columns": speech_text_cols,
        "num_features": len(feature_cols),
        "numerical_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "labeled_phq_rows": int(df['phq9_cat_end'].notnull().sum()),  # pyright: ignore[reportArgumentType]
        "unlabeled_phq_rows": int(df['phq9_cat_end'].isnull().sum())  # pyright: ignore[reportArgumentType]
    }
    
    return feature_cols, num_cols, cat_cols, target_cols, inspection_summary

def get_missing_stats(df: pd.DataFrame, feature_cols: list) -> tuple[dict, dict]:
    """
    Computes percentage of missing values per feature.
    """
    missing_series = df[feature_cols].isnull().mean() * 100.0
    missing_dict = missing_series.to_dict()
    
    tier_counts = {
        "tier1_complete_0pct": int(np.sum(missing_series == 0)),
        "tier2_high_quality_under_5pct": int(np.sum(missing_series < 5)),
        "tier3_moderate_5_to_30pct": int(np.sum((missing_series >= 5) & (missing_series < 30))),
        "tier4_sparse_30_to_70pct": int(np.sum((missing_series >= 30) & (missing_series < 70))),
        "tier5_very_sparse_over_70pct": int(np.sum(missing_series >= 70))
    }
    return missing_dict, tier_counts

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    missing_dict, tiers = get_missing_stats(df, f_cols)
    print("Dataset Inspection Summary:")
    print(json.dumps(summary, indent=2))
    print("\nMissingness Tiers:")
    print(json.dumps(tiers, indent=2))
