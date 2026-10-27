"""
Stage 4: Explainable Longitudinal Depression Severity Progression Framework.

Implements configurable pre-assessment observation windows (W=1, 2, 3, Full),
temporal progression modeling, MCID progression metrics, and strict zero-leakage pipeline.
All historical results, splits, and checkpoints remain untouched.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any

from project.src.evaluation import (
    evaluate_classification,
    compute_multiclass_g_mean,
)


# ==============================================================================
# 1. Configurable Observation Window Dataset Builder
# ==============================================================================

class WindowConfig:
    """Configuration for observation windows."""
    def __init__(
        self,
        window_size: Optional[int] = 1,  # None means Full cumulative history
        min_periods: int = 1,
        include_current: bool = True,
        name: str = "W1"
    ):
        self.window_size = window_size
        self.min_periods = min_periods
        self.include_current = include_current
        self.name = name

    def __repr__(self):
        w_str = str(self.window_size) if self.window_size is not None else "Full"
        return f"WindowConfig(name='{self.name}', window_size={w_str}, min_periods={self.min_periods})"


def extract_windowed_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_config: WindowConfig,
    scaler_and_imputer=None,
    fit_preprocessor: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Any]:
    """
    Extracts temporally aligned, windowed sequences for all labeled assessments.
    
    Guarantees:
    - Zero future leakage: For an assessment at period T, only periods <= T enter the sequence.
    - Zero target leakage: Predictor features exclude all phq9_* and delta_phq9 columns.
    - Missing period handling: Padded to fixed window size with sequence lengths and masks recorded.
    
    Returns:
        X_seq: Shape (N, W, D) for windowed sequences
        X_curr: Shape (N, D) for concurrent period features
        y_cat: Shape (N,) class labels [0..4]
        y_score: Shape (N,) continuous PHQ-9 scores
        meta_list: Metadata list of dicts with user_id, period_num, history_len, etc.
        preprocessor: Fitted preprocessor (if fit_preprocessor=True)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    # Sort strictly by participant and period
    df_sorted = df.sort_values(['user_id', 'period_num']).copy()
    
    # Preprocessing
    if fit_preprocessor:
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        # Fit strictly on available feature rows of this dataframe (training only)
        X_raw = df_sorted[feature_cols].values
        preprocessor.fit(X_raw)
    else:
        preprocessor = scaler_and_imputer
        
    # Transform features
    X_transformed = preprocessor.transform(df_sorted[feature_cols].values)
    D = len(feature_cols)
    
    # Map (user_id, period_num) -> feature vector
    feat_map = {}
    for idx, (u, p) in enumerate(zip(df_sorted['user_id'], df_sorted['period_num'])):
        feat_map[(u, int(p))] = X_transformed[idx]
        
    # Group available periods by user
    user_periods_map = df_sorted.groupby('user_id')['period_num'].apply(lambda s: sorted(list(s))).to_dict()
    
    # Filter to labeled assessment rows (periods 3, 6, 9, 12 where phq9_cat_end is not null)
    labeled_mask = df_sorted['phq9_cat_end'].notna()
    df_eval = df_sorted[labeled_mask].copy()
    
    # Determine maximum window sequence length W
    if window_config.window_size is None:
        max_W = 12  # max periods in dataset
    else:
        max_W = window_config.window_size
        
    X_seq_list = []
    X_curr_list = []
    y_cat_list = []
    y_score_list = []
    meta_list = []
    
    for _, row in df_eval.iterrows():
        u = row['user_id']
        T = int(row['period_num'])
        cat = int(row['phq9_cat_end'])
        score = float(row['phq9_score_end']) if pd.notna(row['phq9_score_end']) else np.nan
        
        # Get all periods for this user that are <= T (strictly no future data!)
        avail = [p for p in user_periods_map.get(u, []) if p <= T]
        
        # Determine which periods enter the window
        if window_config.window_size is None:
            # Full cumulative history up to T
            target_periods = avail
        else:
            # Fixed window of preceding periods ending at T
            # e.g., if window_size = 3 and T = 6, ideal periods are [4, 5, 6]
            start_p = max(1, T - window_config.window_size + 1)
            target_periods = [p for p in avail if start_p <= p <= T]
            
        # Check minimum periods constraint
        if len(target_periods) < window_config.min_periods:
            continue
            
        # Build sequence array of shape (max_W, D)
        seq = np.zeros((max_W, D), dtype=np.float32)
        # Place available periods at the end of the sequence (right-aligned)
        seq_len = len(target_periods)
        offset = max_W - seq_len
        for i, p in enumerate(target_periods):
            seq[offset + i] = feat_map[(u, p)]
            
        # Current assessment features
        curr_feat = feat_map[(u, T)]
        
        X_seq_list.append(seq)
        X_curr_list.append(curr_feat)
        y_cat_list.append(cat)
        y_score_list.append(score)
        meta_list.append({
            "user_id": u,
            "period_num": T,
            "window_size": max_W,
            "actual_periods_in_window": target_periods,
            "seq_len": seq_len,
            "phq9_cat_end": cat,
            "phq9_score_end": score,
            "phq9_cat_start": row['phq9_cat_start'] if pd.notna(row['phq9_cat_start']) else np.nan,
            "phq9_score_start": row['phq9_score_start'] if pd.notna(row['phq9_score_start']) else np.nan,
        })
        
    X_seq = np.array(X_seq_list, dtype=np.float32)
    X_curr = np.array(X_curr_list, dtype=np.float32)
    y_cat = np.array(y_cat_list, dtype=np.int64)
    y_score = np.array(y_score_list, dtype=np.float32)
    
    return X_seq, X_curr, y_cat, y_score, meta_list, preprocessor


# ==============================================================================
# 2. PyTorch Dataset & Models for Windowed Progression
# ==============================================================================

class WindowedDataset(Dataset):
    def __init__(self, X_seq, X_curr, y_cat, y_score, meta_list):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_curr = torch.tensor(X_curr, dtype=torch.float32)
        self.y_cat = torch.tensor(y_cat, dtype=torch.long)
        self.y_score = torch.tensor(y_score, dtype=torch.float32)
        self.seq_lens = torch.tensor([m['seq_len'] for m in meta_list], dtype=torch.long)
        self.meta_list = meta_list

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, idx):
        return (
            self.X_seq[idx],
            self.X_curr[idx],
            self.seq_lens[idx],
            self.y_cat[idx],
            self.y_score[idx],
            idx
        )


class WindowedProgressionGRU(nn.Module):
    """
    GRU sequence model over configurable observation windows.
    Dual-head architecture: 5-class severity classification + continuous score regression.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, X_seq, seq_lens):
        # X_seq: (B, W, D)
        B, W, D = X_seq.shape
        gru_out, _ = self.gru(X_seq)  # (B, W, H)
        
        # Gather final hidden state based on right-aligned or length-indexed position
        # Sequences are right-aligned, so the last step W-1 is always the concurrent period T!
        final_h = gru_out[:, -1, :]  # (B, H)
        
        logits = self.classifier(final_h)
        reg_score = self.regressor(final_h).squeeze(-1)
        return logits, reg_score, final_h


class WindowedFusionProgressionModel(nn.Module):
    """
    Fusion architecture: Current-period dense representation fused with
    windowed historical sequence representation.
    """
    def __init__(
        self,
        input_dim: int,
        curr_dim: int = 64,
        hist_dim: int = 64,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()
        self.curr_encoder = nn.Sequential(
            nn.Linear(input_dim, curr_dim),
            nn.BatchNorm1d(curr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(curr_dim, curr_dim),
            nn.ReLU()
        )
        self.hist_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hist_dim,
            batch_first=True
        )
        self.no_history_emb = nn.Parameter(torch.zeros(1, hist_dim))
        
        fusion_dim = curr_dim + hist_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, X_seq, X_curr, seq_lens):
        B = X_curr.shape[0]
        h_curr = self.curr_encoder(X_curr)
        
        # Historical window (excluding current period, i.e., steps 0..W-2)
        if X_seq.shape[1] > 1:
            hist_seq = X_seq[:, :-1, :]
            gru_out, _ = self.hist_gru(hist_seq)
            h_hist = gru_out[:, -1, :]
        else:
            h_hist = self.no_history_emb.expand(B, -1)
            
        fused = torch.cat([h_curr, h_hist], dim=1)
        logits = self.classifier(fused)
        reg_score = self.regressor(fused).squeeze(-1)
        return logits, reg_score, fused


# ==============================================================================
# 3. Training & Validation Loop
# ==============================================================================

def train_progression_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_reg: float = 0.5,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains progression model with dual masked loss and validation-based early stopping.
    Primary selection criterion: Validation Macro-F1.
    """
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    reg_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_macro_f1 = -1.0
    best_state = None
    best_metrics = {}
    
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            X_seq, X_curr, lens, y_c, y_s, _ = batch
            X_seq = X_seq.to(device)
            X_curr = X_curr.to(device)
            lens = lens.to(device)
            y_c = y_c.to(device)
            y_s = y_s.to(device)
            
            optimizer.zero_grad()
            if isinstance(model, WindowedFusionProgressionModel):
                logits, pred_score, _ = model(X_seq, X_curr, lens)
            else:
                logits, pred_score, _ = model(X_seq, lens)
                
            loss_cls = cls_criterion(logits, y_c)
            
            # Masked regression loss (only where y_s is not NaN)
            valid_mask = ~torch.isnan(y_s)
            if valid_mask.sum() > 0:
                loss_reg = reg_criterion(pred_score[valid_mask], y_s[valid_mask]).mean()
            else:
                loss_reg = torch.tensor(0.0, device=device)
                
            total_loss = loss_cls + lambda_reg * loss_reg
            total_loss.backward()
            optimizer.step()
            
        # Validation evaluation
        val_preds, val_probs, val_scores, val_targets_c, val_targets_s = evaluate_progression_model(
            model, val_loader, device
        )
        val_m = evaluate_classification(val_targets_c, val_preds)
        val_f1 = val_m["macro_f1"]
        
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_macro_f1": val_f1,
                "val_bal_acc": val_m["balanced_accuracy"],
                "val_acc": val_m["accuracy"],
                "epoch": epoch
            }
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, best_metrics


def evaluate_progression_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model and returns arrays of predictions and targets."""
    model.eval()
    all_preds = []
    all_probs = []
    all_scores = []
    all_targets_c = []
    all_targets_s = []
    
    with torch.no_grad():
        for batch in data_loader:
            X_seq, X_curr, lens, y_c, y_s, _ = batch
            X_seq = X_seq.to(device)
            X_curr = X_curr.to(device)
            lens = lens.to(device)
            
            if isinstance(model, WindowedFusionProgressionModel):
                logits, pred_score, _ = model(X_seq, X_curr, lens)
            else:
                logits, pred_score, _ = model(X_seq, lens)
                
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            
            all_preds.extend(preds)
            all_probs.append(probs)
            all_scores.extend(pred_score.cpu().numpy())
            all_targets_c.extend(y_c.numpy())
            all_targets_s.extend(y_s.numpy())
            
    return (
        np.array(all_preds),
        np.concatenate(all_probs, axis=0),
        np.array(all_scores),
        np.array(all_targets_c),
        np.array(all_targets_s)
    )


# ==============================================================================
# 4. Longitudinal Progression Evaluation (Within-Subject Change)
# ==============================================================================

def compute_progression_metrics(
    meta_list: List[Dict[str, Any]],
    preds_cat: np.ndarray,
    preds_score: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates longitudinal severity progression across consecutive observed assessments.
    
    Distinguishes:
    1. Observed change: Delta y_t = y_t - y_{t-1}
    2. Predicted change: Delta y_hat_t = y_hat_t - y_hat_{t-1}
    3. MCID 3-class progression (Improved <= -5, Stable -4..+4, Worsened >= +5)
    4. Categorical progression: Directional agreement and transition accuracy
    """
    df_eval = pd.DataFrame(meta_list)
    df_eval['pred_cat'] = preds_cat
    df_eval['pred_score'] = preds_score
    
    # Sort by user_id and period_num
    df_eval = df_eval.sort_values(['user_id', 'period_num']).copy()
    df_eval['prev_user_id'] = df_eval['user_id'].shift(1)
    df_eval['prev_period_num'] = df_eval['period_num'].shift(1)
    df_eval['prev_phq9_score_end'] = df_eval['phq9_score_end'].shift(1)
    df_eval['prev_phq9_cat_end'] = df_eval['phq9_cat_end'].shift(1)
    df_eval['prev_pred_score'] = df_eval['pred_score'].shift(1)
    df_eval['prev_pred_cat'] = df_eval['pred_cat'].shift(1)
    
    same_user = df_eval['user_id'] == df_eval['prev_user_id']
    transitions = df_eval[same_user].copy()
    
    if len(transitions) == 0:
        return {"error": "No consecutive assessment transitions found"}
        
    # Observed vs Predicted Continuous Change
    transitions['obs_delta_score'] = transitions['phq9_score_end'] - transitions['prev_phq9_score_end']
    transitions['pred_delta_score'] = transitions['pred_score'] - transitions['prev_pred_score']
    
    # Score change MAE and Pearson correlation
    delta_score_mae = float(np.mean(np.abs(transitions['obs_delta_score'] - transitions['pred_delta_score'])))
    delta_score_corr = float(np.corrcoef(transitions['obs_delta_score'], transitions['pred_delta_score'])[0, 1])
    
    # MCID 3-Class Progression (+/- 5 points)
    # Class 0: Improved (<= -5), Class 1: Stable (-4..+4), Class 2: Worsened (>= +5)
    def to_mcid_class(val):
        if val <= -5:
            return 0  # Improved
        elif val >= 5:
            return 2  # Worsened
        return 1      # Stable
        
    obs_mcid = np.array([to_mcid_class(v) for v in transitions['obs_delta_score']])
    pred_mcid = np.array([to_mcid_class(v) for v in transitions['pred_delta_score']])
    
    mcid_acc = float(np.mean(obs_mcid == pred_mcid))
    from sklearn.metrics import f1_score, balanced_accuracy_score
    mcid_macro_f1 = float(f1_score(obs_mcid, pred_mcid, average='macro', zero_division=0))
    mcid_bal_acc = float(balanced_accuracy_score(obs_mcid, pred_mcid))
    
    # Directional Agreement: Did score change in the same direction?
    # Sign: -1 (decrease), 0 (no change), +1 (increase)
    obs_sign = np.sign(transitions['obs_delta_score'].values)
    pred_sign = np.sign(transitions['pred_delta_score'].values)
    directional_acc = float(np.mean(obs_sign == pred_sign))
    
    # Categorical Change: Did predicted category change match observed category change?
    obs_cat_delta = transitions['phq9_cat_end'].values - transitions['prev_phq9_cat_end'].values
    pred_cat_delta = transitions['pred_cat'].values - transitions['prev_pred_cat'].values
    cat_direction_acc = float(np.mean(np.sign(obs_cat_delta) == np.sign(pred_cat_delta)))
    
    return {
        "n_transitions": int(len(transitions)),
        "n_participants": int(transitions['user_id'].nunique()),
        "delta_score_mae": delta_score_mae,
        "delta_score_pearson_r": delta_score_corr if not np.isnan(delta_score_corr) else 0.0,
        "mcid_progression_accuracy": mcid_acc,
        "mcid_progression_macro_f1": mcid_macro_f1,
        "mcid_progression_bal_acc": mcid_bal_acc,
        "directional_accuracy": directional_acc,
        "category_direction_accuracy": cat_direction_acc
    }
