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

class ClassificationMLP(nn.Module):
    """
    Standard clean classification-only MLP for 5-class depression severity prediction.
    Optionally supports an auxiliary continuous score regression head for controlled ablations.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 5, dropout: float = 0.2, include_regressor: bool = True):
        super(ClassificationMLP, self).__init__()
        self.include_regressor = include_regressor
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
        if include_regressor:
            self.regressor = nn.Linear(hidden_dim // 2, 1)
        else:
            self.regressor = None

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        if self.regressor is not None:
            reg_out = self.regressor(features).squeeze(-1)
        else:
            reg_out = None
        return logits, reg_out, features

def prepare_dataset_arrays(df: pd.DataFrame, preprocessor: DataPreprocessor, fit_preprocessor: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extracts features and targets for labeled rows (phq9_cat_end not null).
    Guarantees preprocessing is fitted strictly on the provided DataFrame when requested (TRAIN only).
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

def compute_class_weights(y_train_cat: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Computes balanced class weights strictly from training labels:
    w_c = N / (C * N_c)
    """
    classes, counts = np.unique(y_train_cat, return_counts=True)
    total_samples = len(y_train_cat)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c, count in zip(classes, counts):
        weights[c] = total_samples / (num_classes * count)
    return torch.tensor(weights, dtype=torch.float32)

def train_baseline_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    use_class_weights: bool = False,
    aux_reg_weight: float = 0.5,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 128,
    patience: int = 6,
    seed: int = 42,
    evaluate_test: bool = True,
    checkpoint_name: str = "baseline_classification_best.pt"
) -> tuple[ClassificationMLP, DataPreprocessor, dict, tuple, tuple]:
    """
    Trains an MLP with:
    1. Validation Macro-F1 checkpoint selection (Balanced Accuracy tie-breaker).
    2. Patience-based early stopping.
    3. Optional class weighting (computed strictly from train labels).
    4. Optional auxiliary regression loss weight (0.5 by default for joint training, or 0.0 for pure classification).
    """
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
    
    include_regressor = True
    model = ClassificationMLP(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        num_classes=5,
        dropout=0.2,
        include_regressor=include_regressor
    )
    
    if use_class_weights:
        class_weights = compute_class_weights(y_train_cat, num_classes=5)
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion_cls = nn.CrossEntropyLoss()
        
    criterion_reg = nn.MSELoss() if include_regressor else None
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    best_epoch = 0
    best_model_weights = None
    patience_counter = 0
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_c, y_s in train_loader:
            optimizer.zero_grad()
            logits, reg_out, _ = model(x_b)
            loss_c = criterion_cls(logits, y_c)
            if include_regressor and reg_out is not None and criterion_reg is not None:
                loss_r = criterion_reg(reg_out, y_s)
                loss = loss_c + aux_reg_weight * loss_r
            else:
                loss = loss_c
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= len(train_dataset)
        
        # Validation phase: evaluate strictly with classification metrics
        model.eval()
        val_preds_cls = []
        val_preds_reg = []
        with torch.no_grad():
            for x_b, y_c, y_s in val_loader:
                logits, reg_out, _ = model(x_b)
                val_preds_cls.extend(torch.argmax(logits, dim=1).cpu().numpy())
                if reg_out is not None:
                    val_preds_reg.extend(reg_out.cpu().numpy())
                    
        val_cls = evaluate_classification(y_val_cat, np.array(val_preds_cls))
        val_macro_f1 = val_cls["macro_f1"]
        val_bal_acc = val_cls["balanced_accuracy"]
        val_acc = val_cls["accuracy"]
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_macro_f1,
            "val_bal_acc": val_bal_acc,
            "val_acc": val_acc
        })
        
        # Checkpoint selection strictly on Validation Macro-F1 (tie breaker: balanced accuracy)
        is_best = False
        if val_macro_f1 > best_val_macro_f1 + 1e-5:
            is_best = True
        elif abs(val_macro_f1 - best_val_macro_f1) <= 1e-5 and val_bal_acc > best_val_bal_acc:
            is_best = True
            
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Restore best validation model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    # Save checkpoint
    ckpt_path = f"d:/projects/DepDet/project/checkpoints/{checkpoint_name}"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    
    # Final Validation evaluation
    model.eval()
    with torch.no_grad():
        val_logits, val_reg, val_lat = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred_cat = torch.argmax(val_logits, dim=1).numpy()
        val_cls_metrics = evaluate_classification(y_val_cat, val_pred_cat)
        
        if val_reg is not None:
            val_reg_metrics = evaluate_regression(y_val_score, val_reg.numpy())
        else:
            val_reg_metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
            
        if evaluate_test:
            test_logits, test_reg, test_lat = model(torch.tensor(X_test, dtype=torch.float32))
            test_pred_cat = torch.argmax(test_logits, dim=1).numpy()
            test_cls_metrics = evaluate_classification(y_test_cat, test_pred_cat)
            test_reg_metrics = evaluate_regression(y_test_score, test_reg.numpy()) if test_reg is not None else {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
            test_lat_np = test_lat.numpy()
        else:
            test_cls_metrics = None
            test_reg_metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
            test_lat_np = None
            
    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_bal_acc": best_val_bal_acc,
        "val_classification": val_cls_metrics,
        "val_regression": val_reg_metrics,
        "test_classification": test_cls_metrics,
        "test_regression": test_reg_metrics,
        "history": history
    }
    
    return model, prep, metrics, (X_val, y_val_cat, val_lat.numpy()), (X_test, y_test_cat, test_lat_np)

# Backwards compatibility aliases
BaselineMLP = ClassificationMLP
train_baseline = train_baseline_mlp

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    m, p, res, _, _ = train_baseline_mlp(train_df, val_df, test_df, f_cols)
    print("Baseline Classification-Only MLP Results:")
    print(f"Best Val Macro-F1: {res['best_val_macro_f1']:.4f}, Test Macro-F1: {res['test_classification']['macro_f1']:.4f}")
