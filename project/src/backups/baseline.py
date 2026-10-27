import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import copy
from data_loader import load_raw_dataset, inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from evaluation import evaluate_classification, evaluate_regression, plot_confusion_matrix, save_metrics_to_json

class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 5, dropout: float = 0.2):
        super(BaselineMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        reg_out = self.regressor(features).squeeze(-1)
        return logits, reg_out, features

def prepare_dataset_arrays(df: pd.DataFrame, preprocessor: DataPreprocessor, fit_preprocessor: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extracts features and targets for labeled rows (phq9_cat_end not null).
    """
    labeled_df = df[df['phq9_cat_end'].notnull()].copy()
    assert isinstance(labeled_df, pd.DataFrame)
    if fit_preprocessor:
        preprocessor.fit_train(labeled_df)
    
    X = preprocessor.transform(labeled_df)
    y_cat = np.asarray(labeled_df['phq9_cat_end'].astype(int).values)
    y_score = np.asarray(labeled_df['phq9_score_end'].astype(float).values)
    y_delta = np.asarray((labeled_df['phq9_score_end'] - labeled_df['phq9_score_start']).astype(float).values)
    return X, y_cat, y_score, y_delta, labeled_df

def train_baseline(
    train_df, val_df, test_df, feature_cols,
    epochs=15, lr=1e-3, batch_size=128, patience=4, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prep = DataPreprocessor(feature_cols=feature_cols)
    X_train, y_train_cat, y_train_score, _, _ = prepare_dataset_arrays(train_df, prep, fit_preprocessor=True)
    X_val, y_val_cat, y_val_score, _, _ = prepare_dataset_arrays(val_df, prep, fit_preprocessor=False)
    X_test, y_test_cat, y_test_score, _, _ = prepare_dataset_arrays(test_df, prep, fit_preprocessor=False)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_cat, dtype=torch.long),
        torch.tensor(y_train_score, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_cat, dtype=torch.long),
        torch.tensor(y_val_score, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = BaselineMLP(input_dim=X_train.shape[1], hidden_dim=128, num_classes=5)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_c, y_s in train_loader:
            optimizer.zero_grad()
            logits, reg_out, _ = model(x_b)
            loss_c = criterion_cls(logits, y_c)
            loss_r = criterion_reg(reg_out, y_s)
            loss = loss_c + 0.5 * loss_r
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds_cls = []
        with torch.no_grad():
            for x_b, y_c, y_s in val_loader:
                logits, reg_out, _ = model(x_b)
                loss_c = criterion_cls(logits, y_c)
                loss_r = criterion_reg(reg_out, y_s)
                loss = loss_c + 0.5 * loss_r
                val_loss += loss.item() * len(x_b)
                val_preds_cls.extend(torch.argmax(logits, dim=1).numpy())
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping baseline at epoch {epoch}")
                break
                
    # Load best validation model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    # Save checkpoint
    ckpt_path = "d:/projects/DepDet/project/checkpoints/baseline_best.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    
    # Evaluate on Validation
    model.eval()
    with torch.no_grad():
        val_logits, val_reg, val_lat = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred_cat = torch.argmax(val_logits, dim=1).numpy()
        val_pred_score = val_reg.numpy()
        
        test_logits, test_reg, test_lat = model(torch.tensor(X_test, dtype=torch.float32))
        test_pred_cat = torch.argmax(test_logits, dim=1).numpy()
        test_pred_score = test_reg.numpy()
        
    val_cls_metrics = evaluate_classification(y_val_cat, val_pred_cat)
    val_reg_metrics = evaluate_regression(y_val_score, val_pred_score)
    
    test_cls_metrics = evaluate_classification(y_test_cat, test_pred_cat)
    test_reg_metrics = evaluate_regression(y_test_score, test_pred_score)
    
    metrics = {
        "val_classification": val_cls_metrics,
        "val_regression": val_reg_metrics,
        "test_classification": test_cls_metrics,
        "test_regression": test_reg_metrics
    }
    
    save_metrics_to_json(metrics, "d:/projects/DepDet/project/results/metrics/baseline_metrics.json")
    
    class_names = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]
    plot_confusion_matrix(np.array(test_cls_metrics["confusion_matrix"]), class_names,
                          "Baseline MLP Confusion Matrix (Test)",
                          "d:/projects/DepDet/project/results/confusion_matrices/baseline_cm.png")
                          
    return model, prep, metrics, (X_val, y_val_cat, val_lat.numpy()), (X_test, y_test_cat, test_lat.numpy())

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    train_baseline(train_df, val_df, test_df, f_cols)
