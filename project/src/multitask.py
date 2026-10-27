import torch
import torch.nn as nn
import torch.nn.functional as F
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
from baseline import prepare_dataset_arrays
from contrastive import SupConLoss
from oversampling import latent_space_smote
from longitudinal import build_longitudinal_sequences, LongitudinalDataset, collate_seq_batch
from evaluation import evaluate_classification, evaluate_regression, save_metrics_to_json, plot_confusion_matrix

class MultiTaskFullModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        num_classes: int = 5
    ):
        super(MultiTaskFullModel, self).__init__()
        
        # Modality Sub-encoders (split input features into sub-domains or shared encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.hidden_dim = hidden_dim
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
        
        # Multimodal & Longitudinal Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Projection head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Multi-task Prediction Heads
        self.severity_head = nn.Linear(hidden_dim // 2, num_classes)
        self.phq_head = nn.Linear(hidden_dim // 2, 1)
        self.change_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, curr_X, history_X, lengths, has_history=None):
        batch_size = curr_X.shape[0]
        device = curr_X.device
        
        curr_feat = self.encoder(curr_X) # (B, H/2)
        
        gru_out, _ = self.history_gru(history_X)
        
        # Variable-length indexing: strictly extract state at true sequence length (lengths - 1)
        # Instances with zero history receive the learned no_history_emb
        hist_feat = torch.zeros(batch_size, self.hidden_dim // 2, device=device)
        for b in range(batch_size):
            L = lengths[b].item()
            if L > 0:
                hist_feat[b] = gru_out[b, L - 1, :]
            else:
                hist_feat[b] = self.no_history_emb
        
        fused_raw = torch.cat([curr_feat, hist_feat], dim=1) # (B, H)
        patient_state = self.fusion_layer(fused_raw) # (B, H/2)
        
        proj = F.normalize(self.projection_head(patient_state), dim=1)
        
        logits_sev = self.severity_head(patient_state)
        pred_phq = self.phq_head(patient_state).squeeze(-1)
        pred_change = self.change_head(patient_state).squeeze(-1)
        
        return logits_sev, pred_phq, pred_change, proj, patient_state

def train_multitask_full_model(
    train_df, val_df, test_df, feature_cols,
    lambda1=1.0, lambda2=0.5, lambda3=0.5, lambda4=0.2,
    epochs=15, lr=1e-3, batch_size=128, patience=4, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prep = DataPreprocessor(feature_cols=feature_cols)
    prep.fit_train(train_df[train_df['phq9_cat_end'].notnull()])
    
    train_samples = build_longitudinal_sequences(train_df, prep)
    val_samples = build_longitudinal_sequences(val_df, prep)
    test_samples = build_longitudinal_sequences(test_df, prep)
    
    train_loader = DataLoader(LongitudinalDataset(train_samples), batch_size=batch_size, shuffle=True, collate_fn=collate_seq_batch)
    val_loader = DataLoader(LongitudinalDataset(val_samples), batch_size=batch_size, shuffle=False, collate_fn=collate_seq_batch)
    test_loader = DataLoader(LongitudinalDataset(test_samples), batch_size=batch_size, shuffle=False, collate_fn=collate_seq_batch)
    
    input_dim = len(feature_cols)
    model = MultiTaskFullModel(input_dim=input_dim, hidden_dim=128, proj_dim=64, num_classes=5)
    
    criterion_sev = nn.CrossEntropyLoss()
    criterion_phq = nn.MSELoss()
    criterion_change = nn.MSELoss()
    criterion_supcon = SupConLoss(temperature=0.07)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits_sev, pred_phq, pred_change, proj, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"])
            
            l_sev = criterion_sev(logits_sev, batch["y_cat"])
            l_phq = criterion_phq(pred_phq, batch["y_score"])
            l_change = criterion_change(pred_change, batch["y_delta"])
            l_con = criterion_supcon(proj, batch["y_cat"])
            
            loss = lambda1 * l_sev + lambda2 * l_phq + lambda3 * l_change + lambda4 * l_con
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch["y_cat"])
            
        train_loss /= len(train_samples)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits_sev, pred_phq, pred_change, proj, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"])
                l_sev = criterion_sev(logits_sev, batch["y_cat"])
                l_phq = criterion_phq(pred_phq, batch["y_score"])
                l_change = criterion_change(pred_change, batch["y_delta"])
                l_con = criterion_supcon(proj, batch["y_cat"])
                loss = lambda1 * l_sev + lambda2 * l_phq + lambda3 * l_change + lambda4 * l_con
                val_loss += loss.item() * len(batch["y_cat"])
        val_loss /= len(val_samples)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping full model at epoch {epoch}")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # Save Checkpoint
    ckpt_path = "d:/projects/DepDet/project/checkpoints/full_model_best.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    
    # Evaluate on Validation and Test
    model.eval()
    def evaluate_set(loader):
        preds_c, trues_c = [], []
        preds_s, trues_s = [], []
        preds_d, trues_d = [], []
        with torch.no_grad():
            for batch in loader:
                logits_sev, pred_phq, pred_change, _, _ = model(batch["curr_X"], batch["history_X"], batch["lengths"])
                preds_c.extend(torch.argmax(logits_sev, dim=1).numpy())
                trues_c.extend(batch["y_cat"].numpy())
                preds_s.extend(pred_phq.numpy())
                trues_s.extend(batch["y_score"].numpy())
                preds_d.extend(pred_change.numpy())
                trues_d.extend(batch["y_delta"].numpy())
                
        cls_m = evaluate_classification(np.array(trues_c), np.array(preds_c))
        reg_s_m = evaluate_regression(np.array(trues_s), np.array(preds_s))
        reg_d_m = evaluate_regression(np.array(trues_d), np.array(preds_d))
        return cls_m, reg_s_m, reg_d_m
        
    val_cls, val_reg_s, val_reg_d = evaluate_set(val_loader)
    test_cls, test_reg_s, test_reg_d = evaluate_set(test_loader)
    
    metrics = {
        "loss_weights": {"lambda1": lambda1, "lambda2": lambda2, "lambda3": lambda3, "lambda4": lambda4},
        "val_classification": val_cls,
        "val_regression_score": val_reg_s,
        "val_regression_delta": val_reg_d,
        "test_classification": test_cls,
        "test_regression_score": test_reg_s,
        "test_regression_delta": test_reg_d
    }
    
    save_metrics_to_json(metrics, "d:/projects/DepDet/project/results/metrics/full_model_metrics.json")
    
    class_names = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]
    plot_confusion_matrix(np.array(test_cls["confusion_matrix"]), class_names,
                          "Full Proposed Model Confusion Matrix (Test)",
                          "d:/projects/DepDet/project/results/confusion_matrices/full_model_cm.png")
                          
    return model, prep, metrics

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    train_multitask_full_model(train_df, val_df, test_df, f_cols)
