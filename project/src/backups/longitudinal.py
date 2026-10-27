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

def build_longitudinal_sequences(df, preprocessor, max_seq_len=6):
    """
    Constructs per-participant sequence datasets sorted by period_num.
    For each assessment at step t with non-null phq9_cat_end:
      - Current observation: X_t
      - History observations: X_{max(1, t-max_seq_len+1) : t-1} (padded if needed)
      - Targets at step t: phq9_cat_end, phq9_score_end, delta_phq9
      - Future target (t+1 score if available)
    """
    df_sorted = df.sort_values(by=['user_id', 'period_num']).copy()
    
    # Preprocess feature values across entire dataframe
    X_all = preprocessor.transform(df_sorted)
    df_sorted['seq_idx'] = np.arange(len(df_sorted))
    
    user_groups = df_sorted.groupby('user_id')
    
    samples = []
    
    for user_id, group in user_groups:
        group_rows = group.to_dict('records')
        group_X = X_all[group['seq_idx'].values]
        
        for i, row in enumerate(group_rows):
            # Only generate sample if target at step t is labeled
            if pd.isnull(row['phq9_cat_end']):
                continue
                
            curr_X = group_X[i]
            
            # Extract history (previous steps < i)
            if i == 0:
                # No history available, pad with current observation
                history_X = np.expand_dims(curr_X, axis=0) # shape (1, D)
            else:
                start_i = max(0, i - max_seq_len)
                history_X = group_X[start_i:i] # shape (seq_len, D)
                
            y_cat = int(row['phq9_cat_end'])
            y_score = float(row['phq9_score_end'])
            y_delta = float(row['delta_phq9'])
            
            # Check if t+1 assessment score exists
            if i + 1 < len(group_rows) and not pd.isnull(group_rows[i+1]['phq9_score_end']):
                y_next_score = float(group_rows[i+1]['phq9_score_end'])
                has_next = 1.0
            else:
                y_next_score = y_score
                has_next = 0.0
                
            samples.append({
                "user_id": user_id,
                "period_num": row['period_num'],
                "curr_X": curr_X,
                "history_X": history_X,
                "y_cat": y_cat,
                "y_score": y_score,
                "y_delta": y_delta,
                "y_next_score": y_next_score,
                "has_next": has_next
            })
            
    return samples

class LongitudinalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "curr_X": torch.tensor(s["curr_X"], dtype=torch.float32),
            "history_X": s["history_X"], # variable length list/array
            "y_cat": torch.tensor(s["y_cat"], dtype=torch.long),
            "y_score": torch.tensor(s["y_score"], dtype=torch.float32),
            "y_delta": torch.tensor(s["y_delta"], dtype=torch.float32),
            "y_next_score": torch.tensor(s["y_next_score"], dtype=torch.float32),
            "has_next": torch.tensor(s["has_next"], dtype=torch.float32)
        }

def collate_seq_batch(batch):
    curr_X = torch.stack([item["curr_X"] for item in batch])
    y_cat = torch.stack([item["y_cat"] for item in batch])
    y_score = torch.stack([item["y_score"] for item in batch])
    y_delta = torch.stack([item["y_delta"] for item in batch])
    y_next_score = torch.stack([item["y_next_score"] for item in batch])
    has_next = torch.stack([item["has_next"] for item in batch])
    
    # Pad history sequences
    histories = [torch.tensor(item["history_X"], dtype=torch.float32) for item in batch]
    lens = [h.shape[0] for h in histories]
    max_len = max(lens)
    dim = histories[0].shape[1]
    
    padded_history = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    for i, h in enumerate(histories):
        padded_history[i, :h.shape[0], :] = h
        
    lengths = torch.tensor(lens, dtype=torch.long)
    
    return {
        "curr_X": curr_X,
        "history_X": padded_history,
        "lengths": lengths,
        "y_cat": y_cat,
        "y_score": y_score,
        "y_delta": y_delta,
        "y_next_score": y_next_score,
        "has_next": has_next
    }

class LongitudinalGRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 5, mode="current_plus_history"):
        super(LongitudinalGRUModel, self).__init__()
        self.mode = mode # 'current_only', 'history_only', 'current_plus_history'
        
        # Feature Encoder for single observation
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
        
        # Fusion Representation
        if mode == "current_plus_history":
            fusion_dim = (hidden_dim // 2) * 2
        else:
            fusion_dim = hidden_dim // 2
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Prediction Heads
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.reg_score = nn.Linear(hidden_dim // 2, 1)
        self.reg_delta = nn.Linear(hidden_dim // 2, 1)
        self.reg_next_score = nn.Linear(hidden_dim // 2, 1)

    def forward(self, curr_X, history_X, lengths):
        # Current state
        curr_state = self.current_encoder(curr_X) # shape (B, H/2)
        
        # History state
        gru_out, h_n = self.history_gru(history_X) # h_n shape (1, B, H/2)
        history_state = h_n.squeeze(0) # shape (B, H/2)
        
        if self.mode == "current_only":
            fused = curr_state
        elif self.mode == "history_only":
            fused = history_state
        else:
            fused_cat = torch.cat([curr_state, history_state], dim=1)
            fused = self.fusion_layer(fused_cat)
            
        logits = self.classifier(fused)
        score_pred = self.reg_score(fused).squeeze(-1)
        delta_pred = self.reg_delta(fused).squeeze(-1)
        next_score_pred = self.reg_next_score(fused).squeeze(-1)
        
        return logits, score_pred, delta_pred, next_score_pred, fused

def train_longitudinal_experiment(train_df, val_df, test_df, feature_cols, mode="current_plus_history", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prep = DataPreprocessor(feature_cols=feature_cols)
    prep.fit_train(train_df[train_df['phq9_cat_end'].notnull()])
    
    train_samples = build_longitudinal_sequences(train_df, prep)
    val_samples = build_longitudinal_sequences(val_df, prep)
    test_samples = build_longitudinal_sequences(test_df, prep)
    
    train_loader = DataLoader(LongitudinalDataset(train_samples), batch_size=128, shuffle=True, collate_fn=collate_seq_batch)
    val_loader = DataLoader(LongitudinalDataset(val_samples), batch_size=128, shuffle=False, collate_fn=collate_seq_batch)
    test_loader = DataLoader(LongitudinalDataset(test_samples), batch_size=128, shuffle=False, collate_fn=collate_seq_batch)
    
    input_dim = len(feature_cols)
    model = LongitudinalGRUModel(input_dim=input_dim, hidden_dim=128, num_classes=5, mode=mode)
    
    criterion_c = nn.CrossEntropyLoss()
    criterion_r = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(1, 16):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits, score_pred, delta_pred, _, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"])
            l_c = criterion_c(logits, batch["y_cat"])
            l_s = criterion_r(score_pred, batch["y_score"])
            l_d = criterion_r(delta_pred, batch["y_delta"])
            loss = l_c + 0.5 * l_s + 0.5 * l_d
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits, score_pred, delta_pred, _, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"])
                l_c = criterion_c(logits, batch["y_cat"])
                l_s = criterion_r(score_pred, batch["y_score"])
                l_d = criterion_r(delta_pred, batch["y_delta"])
                loss = l_c + 0.5 * l_s + 0.5 * l_d
                val_loss += loss.item() * len(batch["y_cat"])
        val_loss /= len(val_samples)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # Save checkpoint
    ckpt_path = f"d:/projects/DepDet/project/checkpoints/longitudinal_{mode}.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    
    # Evaluate test
    model.eval()
    test_preds_c, test_trues_c = [], []
    test_preds_s, test_trues_s = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, score_pred, _, _, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"])
            test_preds_c.extend(torch.argmax(logits, dim=1).numpy())
            test_trues_c.extend(batch["y_cat"].numpy())
            test_preds_s.extend(score_pred.numpy())
            test_trues_s.extend(batch["y_score"].numpy())
            
    test_cls_metrics = evaluate_classification(np.array(test_trues_c), np.array(test_preds_c))
    test_reg_metrics = evaluate_regression(np.array(test_trues_s), np.array(test_preds_s))
    
    return {
        "mode": mode,
        "test_classification": test_cls_metrics,
        "test_regression": test_reg_metrics
    }

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    
    modes = ["current_only", "history_only", "current_plus_history"]
    history_results = {}
    for m in modes:
        res = train_longitudinal_experiment(train_df, val_df, test_df, f_cols, mode=m)
        history_results[m] = res
        print(f"History Integration Mode '{m}' -> Test Macro-F1: {res['test_classification']['macro_f1']:.4f}, Test MAE: {res['test_regression']['mae']:.4f}")
        
    save_metrics_to_json(history_results, "d:/projects/DepDet/project/results/metrics/history_integration_comparison.json")
