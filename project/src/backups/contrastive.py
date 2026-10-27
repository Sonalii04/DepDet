import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import json
import os
import copy
from data_loader import load_raw_dataset, inspect_modalities_and_features
from splitting import perform_subject_level_split
from preprocessing import DataPreprocessor
from baseline import prepare_dataset_arrays
from evaluation import evaluate_classification, evaluate_regression, save_metrics_to_json, plot_confusion_matrix

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al., 2020)
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [N, D] (normalized embeddings)
        labels: [N]
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss
        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, proj_dim: int = 64, num_classes: int = 5):
        super(ContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        # Projection Head for SupCon
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        # Prediction Heads
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        features = self.encoder(x)
        proj = F.normalize(self.projection_head(features), dim=1)
        logits = self.classifier(features)
        reg_out = self.regressor(features).squeeze(-1)
        return logits, reg_out, proj, features

def train_contrastive_model(
    train_df, val_df, test_df, feature_cols,
    lambda_contrastive=0.2, epochs=15, lr=1e-3, batch_size=128, patience=4, seed=42
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
    
    model = ContrastiveModel(input_dim=X_train.shape[1], hidden_dim=128, proj_dim=64, num_classes=5)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion_supcon = SupConLoss(temperature=0.07)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_c, y_s in train_loader:
            optimizer.zero_grad()
            logits, reg_out, proj, _ = model(x_b)
            
            loss_c = criterion_cls(logits, y_c)
            loss_r = criterion_reg(reg_out, y_s)
            loss_con = criterion_supcon(proj, y_c)
            
            loss = loss_c + 0.5 * loss_r + lambda_contrastive * loss_con
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
            
        train_loss /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_c, y_s in val_loader:
                logits, reg_out, proj, _ = model(x_b)
                loss_c = criterion_cls(logits, y_c)
                loss_r = criterion_reg(reg_out, y_s)
                loss_con = criterion_supcon(proj, y_c)
                loss = loss_c + 0.5 * loss_r + lambda_contrastive * loss_con
                val_loss += loss.item() * len(x_b)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping contrastive at epoch {epoch}")
                break
                
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    # Save checkpoint
    ckpt_path = "d:/projects/DepDet/project/checkpoints/contrastive_best.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        val_logits, val_reg, _, val_lat = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred_cat = torch.argmax(val_logits, dim=1).numpy()
        val_pred_score = val_reg.numpy()
        
        test_logits, test_reg, _, test_lat = model(torch.tensor(X_test, dtype=torch.float32))
        test_pred_cat = torch.argmax(test_logits, dim=1).numpy()
        test_pred_score = test_reg.numpy()
        
    val_cls_metrics = evaluate_classification(y_val_cat, val_pred_cat)
    val_reg_metrics = evaluate_regression(y_val_score, val_pred_score)
    test_cls_metrics = evaluate_classification(y_test_cat, test_pred_cat)
    test_reg_metrics = evaluate_regression(y_test_score, test_pred_score)
    
    metrics = {
        "lambda_contrastive": lambda_contrastive,
        "val_classification": val_cls_metrics,
        "val_regression": val_reg_metrics,
        "test_classification": test_cls_metrics,
        "test_regression": test_reg_metrics
    }
    
    return model, metrics, test_lat.numpy(), y_test_cat

def visualize_latent_spaces(baseline_lat, contrastive_lat, labels, save_dir="d:/projects/DepDet/project/results/latent_spaces"):
    """
    Visualizes and compares Baseline vs Contrastive latent spaces using PCA, UMAP, and t-SNE.
    """
    os.makedirs(save_dir, exist_ok=True)
    class_names = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]
    palette = sns.color_palette("viridis", n_colors=5)
    
    # Subsample for speed if dataset is large
    n_pts = min(1000, len(labels))
    idx_sub = np.random.choice(len(labels), size=n_pts, replace=False)
    b_sub = baseline_lat[idx_sub]
    c_sub = contrastive_lat[idx_sub]
    l_sub = labels[idx_sub]
    
    methods = [
        ("PCA", PCA(n_components=2)),
        ("t-SNE", TSNE(n_components=2, random_state=42, max_iter=300)),
        ("UMAP", umap.UMAP(n_components=2, random_state=42, n_epochs=100))
    ]
    
    for name, reducer in methods:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fit baseline
        b_2d = reducer.fit_transform(b_sub)
        # Fit contrastive
        c_2d = reducer.fit_transform(c_sub)
        
        for cls_idx in range(5):
            idx = (l_sub == cls_idx)
            axes[0].scatter(b_2d[idx, 0], b_2d[idx, 1], label=class_names[cls_idx], alpha=0.6, s=20, color=palette[cls_idx])
            axes[1].scatter(c_2d[idx, 0], c_2d[idx, 1], label=class_names[cls_idx], alpha=0.6, s=20, color=palette[cls_idx])
            
        axes[0].set_title(f"Baseline Latent Space ({name})", fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        axes[1].set_title(f"Contrastive Latent Space ({name})", fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"latent_space_comparison_{name.lower()}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved latent space comparison figure: {save_path}")

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    
    # Tune lambda on Validation
    best_lambda = 0.2
    best_val_f1 = -1.0
    best_contrastive_metrics = None
    
    for l_val in [0.05, 0.1, 0.2, 0.5]:
        m, metrics, _, _ = train_contrastive_model(train_df, val_df, test_df, f_cols, lambda_contrastive=l_val)
        val_f1 = metrics["val_classification"]["macro_f1"]
        print(f"Lambda {l_val} -> Val Macro-F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_lambda = l_val
            best_contrastive_metrics = metrics
            
    print(f"Selected best lambda_contrastive = {best_lambda}")
    save_metrics_to_json(best_contrastive_metrics, "d:/projects/DepDet/project/results/metrics/contrastive_metrics.json")
