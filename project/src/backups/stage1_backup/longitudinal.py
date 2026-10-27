import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import copy
from data_loader import load_raw_dataset, inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from evaluation import evaluate_classification, evaluate_regression, save_metrics_to_json

def build_longitudinal_sequences(df: pd.DataFrame, preprocessor: DataPreprocessor, max_seq_len: int = 6) -> list[dict]:
    """
    Constructs per-participant sequence datasets sorted strictly by [user_id, period_num].
    For each labeled assessment at step i (period t):
      - Current observation: X_t (allowed concurrent features)
      - History observations: strictly previous steps (index < i). NO future data, NO current duplication.
      - If i == 0 (no history available): history_X is an empty array with length 0, has_history = 0.
    """
    df_sorted = df.sort_values(by=['user_id', 'period_num']).copy()
    
    # Preprocess feature values across entire dataframe using TRAIN-fitted preprocessor
    X_all = preprocessor.transform(df_sorted)
    df_sorted['seq_idx'] = np.arange(len(df_sorted))
    
    user_groups = df_sorted.groupby('user_id')
    samples = []
    
    for user_id, group in user_groups:
        group_rows = group.to_dict('records')
        group_X = X_all[group['seq_idx'].values]
        
        for i, row in enumerate(group_rows):
            # Only generate sample if target at step i is labeled
            if pd.isnull(row['phq9_cat_end']):
                continue
                
            curr_X = group_X[i]
            current_period = row['period_num']
            
            # Extract history strictly from previous steps (indices < i)
            if i == 0:
                # No previous observation exists for this participant
                history_X = np.empty((0, curr_X.shape[0]), dtype=np.float32)
                has_history = 0.0
                history_len = 0
            else:
                start_i = max(0, i - max_seq_len)
                history_X = group_X[start_i:i]  # shape: (history_len, D)
                
                # Assert chronological validity: all history observations must be strictly before current period
                for past_idx in range(start_i, i):
                    past_period = group_rows[past_idx]['period_num']
                    assert past_period < current_period, (
                        f"Temporal violation: past_period {past_period} >= current_period {current_period} for user {user_id}"
                    )
                    
                has_history = 1.0
                history_len = len(history_X)
                
            y_cat = int(row['phq9_cat_end'])
            y_score = float(row['phq9_score_end'])
            y_delta = float(row['phq9_score_end'] - row['phq9_score_start'])
            
            samples.append({
                "user_id": str(user_id),
                "period_num": int(current_period),
                "curr_X": curr_X,
                "history_X": history_X,
                "has_history": has_history,
                "history_len": history_len,
                "y_cat": y_cat,
                "y_score": y_score,
                "y_delta": y_delta
            })
            
    return samples

class LongitudinalDataset(Dataset):
    def __init__(self, samples: list[dict], require_history: bool = False):
        if require_history:
            self.samples = [s for s in samples if s["has_history"] > 0.5]
        else:
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "curr_X": torch.tensor(s["curr_X"], dtype=torch.float32),
            "history_X": s["history_X"],
            "has_history": torch.tensor(s["has_history"], dtype=torch.float32),
            "history_len": s["history_len"],
            "y_cat": torch.tensor(s["y_cat"], dtype=torch.long),
            "y_score": torch.tensor(s["y_score"], dtype=torch.float32),
            "y_delta": torch.tensor(s["y_delta"], dtype=torch.float32)
        }

def collate_seq_batch(batch: list[dict]) -> dict:
    curr_X = torch.stack([item["curr_X"] for item in batch])
    y_cat = torch.stack([item["y_cat"] for item in batch])
    y_score = torch.stack([item["y_score"] for item in batch])
    y_delta = torch.stack([item["y_delta"] for item in batch])
    has_history = torch.stack([item["has_history"] for item in batch])
    
    lens = [item["history_len"] for item in batch]
    max_len = max(max(lens), 1)
    dim = curr_X.shape[1]
    
    padded_history = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    for i, item in enumerate(batch):
        h = item["history_X"]
        if len(h) > 0:
            padded_history[i, :len(h), :] = torch.tensor(h, dtype=torch.float32)
            
    lengths = torch.tensor(lens, dtype=torch.long)
    
    return {
        "curr_X": curr_X,
        "history_X": padded_history,
        "lengths": lengths,
        "has_history": has_history,
        "y_cat": y_cat,
        "y_score": y_score,
        "y_delta": y_delta
    }

class LongitudinalGRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 5, mode: str = "current_plus_history"):
        super(LongitudinalGRUModel, self).__init__()
        assert mode in ["current_only", "history_only", "current_plus_history"], f"Unknown mode: {mode}"
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        # Feature Encoder for current observation
        self.current_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # History Encoder (GRU)
        self.history_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Learned embedding for instances with zero history
        self.no_history_emb = nn.Parameter(torch.zeros(hidden_dim // 2))
        nn.init.normal_(self.no_history_emb, std=0.02)
        
        # History presence indicator projection
        self.history_gate = nn.Linear(1, hidden_dim // 2)
        
        # Fusion Representation
        if mode == "current_plus_history":
            fusion_dim = (hidden_dim // 2) * 2
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU()
            )
        else:
            self.fusion_layer = nn.Identity()
            
        # Prediction Heads (cleaned, no unused dead heads)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.reg_score = nn.Linear(hidden_dim // 2, 1)

    def forward(self, curr_X: torch.Tensor, history_X: torch.Tensor, lengths: torch.Tensor, has_history: torch.Tensor):
        batch_size = curr_X.shape[0]
        device = curr_X.device
        
        # 1. Encode current observation
        curr_state = self.current_encoder(curr_X)  # shape: (B, H/2)
        
        # 2. Encode history with proper masking/indexing so padding steps never affect representation
        gru_out, _ = self.history_gru(history_X)  # shape: (B, max_len, H/2)
        
        # For each sample, extract the hidden state strictly at its true length (lengths - 1)
        # For samples with length == 0, use the learned no_history_emb
        history_state = torch.zeros(batch_size, self.hidden_dim // 2, device=device)
        for b in range(batch_size):
            L = lengths[b].item()
            if L > 0:
                history_state[b] = gru_out[b, L - 1, :]
            else:
                history_state[b] = self.no_history_emb
                
        if self.mode == "current_only":
            fused = curr_state
        elif self.mode == "history_only":
            fused = history_state
        else:
            fused_cat = torch.cat([curr_state, history_state], dim=1)
            fused = self.fusion_layer(fused_cat)
            
        logits = self.classifier(fused)
        score_pred = self.reg_score(fused).squeeze(-1)
        return logits, score_pred, fused

def train_longitudinal_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    mode: str = "current_plus_history",
    epochs: int = 25,
    patience: int = 6,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
    evaluate_test: bool = True
) -> dict:
    """
    Trains longitudinal models with:
    - Zero current-observation duplication in history.
    - True variable-length indexing (padding does not contaminate history).
    - Validation Macro-F1 checkpoint selection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prep = DataPreprocessor(feature_cols=feature_cols)
    prep.fit_train(train_df[train_df['phq9_cat_end'].notnull()])
    
    train_samples = build_longitudinal_sequences(train_df, prep)
    val_samples = build_longitudinal_sequences(val_df, prep)
    test_samples = build_longitudinal_sequences(test_df, prep)
    
    # If mode is history_only, filter to observations with valid history
    require_history = (mode == "history_only")
    train_ds = LongitudinalDataset(train_samples, require_history=require_history)
    val_ds = LongitudinalDataset(val_samples, require_history=require_history)
    test_ds = LongitudinalDataset(test_samples, require_history=require_history)
    
    print(f"[{mode.upper()}] Samples with valid evaluation target -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_seq_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq_batch)
    
    input_dim = len(feature_cols)
    model = LongitudinalGRUModel(input_dim=input_dim, hidden_dim=128, num_classes=5, mode=mode)
    
    criterion_c = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    best_epoch = 0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"], batch["has_history"])
            loss = criterion_c(logits, batch["y_cat"])
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_preds_c, val_trues_c = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, _, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"], batch["has_history"])
                val_preds_c.extend(torch.argmax(logits, dim=1).numpy())
                val_trues_c.extend(batch["y_cat"].numpy())
                
        val_cls = evaluate_classification(np.array(val_trues_c), np.array(val_preds_c))
        val_macro_f1 = val_cls["macro_f1"]
        val_bal_acc = val_cls["balanced_accuracy"]
        
        is_best = False
        if val_macro_f1 > best_val_macro_f1 + 1e-5:
            is_best = True
        elif abs(val_macro_f1 - best_val_macro_f1) <= 1e-5 and val_bal_acc > best_val_bal_acc:
            is_best = True
            
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    ckpt_path = f"d:/projects/DepDet/project/checkpoints/longitudinal_{mode}_corrected.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    
    # Validation metrics
    model.eval()
    val_preds_c, val_trues_c, val_preds_s, val_trues_s = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, score_pred, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"], batch["has_history"])
            val_preds_c.extend(torch.argmax(logits, dim=1).numpy())
            val_trues_c.extend(batch["y_cat"].numpy())
            val_preds_s.extend(score_pred.numpy())
            val_trues_s.extend(batch["y_score"].numpy())
            
    val_cls_metrics = evaluate_classification(np.array(val_trues_c), np.array(val_preds_c))
    val_reg_metrics = evaluate_regression(np.array(val_trues_s), np.array(val_preds_s))
    
    if evaluate_test:
        test_preds_c, test_trues_c, test_preds_s, test_trues_s = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                logits, score_pred, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"], batch["has_history"])
                test_preds_c.extend(torch.argmax(logits, dim=1).numpy())
                test_trues_c.extend(batch["y_cat"].numpy())
                test_preds_s.extend(score_pred.numpy())
                test_trues_s.extend(batch["y_score"].numpy())
        test_cls_metrics = evaluate_classification(np.array(test_trues_c), np.array(test_preds_c))
        test_reg_metrics = evaluate_regression(np.array(test_trues_s), np.array(test_preds_s))
    else:
        test_cls_metrics = None
        test_reg_metrics = None
        
    return {
        "mode": mode,
        "best_epoch": best_epoch,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_test_samples": len(test_ds),
        "val_classification": val_cls_metrics,
        "val_regression": val_reg_metrics,
        "test_classification": test_cls_metrics,
        "test_regression": test_reg_metrics
    }
