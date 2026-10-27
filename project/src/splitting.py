import numpy as np
import pandas as pd
import json
import os

def perform_subject_level_split(df: pd.DataFrame, seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, save_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Performs strict subject-level splitting (70% train, 15% val, 15% test).
    Ensures zero participant overlap across splits.
    Saves split participant IDs to JSON only if save_dir is specified.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    unique_users = np.array(df['user_id'].unique())
    np.random.seed(seed)
    shuffled_users = np.random.permutation(unique_users)
    
    n_users = len(shuffled_users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_users = shuffled_users[:n_train].tolist()
    val_users = shuffled_users[n_train:n_train + n_val].tolist()
    test_users = shuffled_users[n_train + n_val:].tolist()
    
    set_train = set(train_users)
    set_val = set(val_users)
    set_test = set(test_users)
    
    # Assert zero overlap
    assert len(set_train.intersection(set_val)) == 0, "Overlap detected between Train and Val!"
    assert len(set_train.intersection(set_test)) == 0, "Overlap detected between Train and Test!"
    assert len(set_val.intersection(set_test)) == 0, "Overlap detected between Val and Test!"
    
    split_info = {
        "seed": seed,
        "total_participants": n_users,
        "n_train_participants": len(train_users),
        "n_val_participants": len(val_users),
        "n_test_participants": len(test_users),
        "train_user_ids": train_users,
        "val_user_ids": val_users,
        "test_user_ids": test_users
    }
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "participant_splits.json")
        with open(json_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Subject-level split saved to {json_path}")
        
    print(f"Participants -> Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
    
    train_df = df[df['user_id'].isin(train_users)].copy()
    val_df = df[df['user_id'].isin(val_users)].copy()
    test_df = df[df['user_id'].isin(test_users)].copy()
    
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    
    print(f"Rows -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df, split_info

if __name__ == "__main__":
    from data_loader import load_raw_dataset
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
