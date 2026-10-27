import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_train(self, train_df: pd.DataFrame) -> None:
        """
        Fits imputer and scaler ONLY on the training split.
        """
        X_train_raw = train_df[self.feature_cols].values
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        self.scaler.fit(X_train_imp)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame using pre-fitted imputer and scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on TRAIN first!")
        X_raw = df[self.feature_cols].values
        X_imp = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imp)
        assert isinstance(X_scaled, np.ndarray)
        return X_scaled

def save_preprocessor(preprocessor: DataPreprocessor, path: str = "d:/projects/DepDet/project/checkpoints/preprocessor.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Saved preprocessor object to {path}")

def load_preprocessor(path: str = "d:/projects/DepDet/project/checkpoints/preprocessor.joblib") -> DataPreprocessor:
    return joblib.load(path)

def get_split_reporting_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Computes comprehensive split reporting stats requested by Section 2.
    """
    def get_stats(df: pd.DataFrame, name: str) -> dict:
        labeled = df[df['phq9_cat_end'].notnull()]
        assert isinstance(labeled, pd.DataFrame)
        class_dist = labeled['phq9_cat_end'].value_counts().to_dict()
        return {
            "split": name,
            "total_rows": len(df),
            "num_participants": int(df['user_id'].nunique()),  # pyright: ignore[reportArgumentType]
            "labeled_rows": len(labeled),
            "unlabeled_rows": len(df) - len(labeled),
            "class_distribution_labeled": {int(k): int(v) for k, v in class_dist.items()}  # pyright: ignore[reportAttributeAccessIssue]
        }

    return {
        "train": get_stats(train_df, "train"),
        "val": get_stats(val_df, "val"),
        "test": get_stats(test_df, "test")
    }

if __name__ == "__main__":
    from data_loader import load_raw_dataset, inspect_modalities_and_features
    from splitting import perform_subject_level_split
    
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, _ = perform_subject_level_split(df)
    
    prep = DataPreprocessor(feature_cols=f_cols)
    prep.fit_train(train_df)
    save_preprocessor(prep)
    
    X_train = prep.transform(train_df)
    X_val = prep.transform(val_df)
    X_test = prep.transform(test_df)
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
