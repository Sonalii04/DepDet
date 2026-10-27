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
from baseline import prepare_dataset_arrays
from evaluation import evaluate_classification, evaluate_regression, save_metrics_to_json

def latent_space_smote(latent_embeddings, labels, max_samples_per_class=None):
    """
    Performs Latent-space SMOTE / interpolation ONLY on training embeddings:
    z_new = z_i + alpha * (z_j - z_i), alpha ~ Uniform(0, 1)
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    if max_samples_per_class is None:
        target_count = np.max(counts)
    else:
        target_count = max_samples_per_class
        
    synth_latents = [latent_embeddings]
    synth_labels = [labels]
    
    for cls in unique_classes:
        cls_mask = (labels == cls)
        cls_latents = latent_embeddings[cls_mask]
        n_existing = len(cls_latents)
        n_needed = target_count - n_existing
        
        if n_needed <= 0 or n_existing < 2:
            continue
            
        new_samples = []
        for _ in range(n_needed):
            idx1, idx2 = np.random.choice(n_existing, size=2, replace=True)
            z_i = cls_latents[idx1]
            z_j = cls_latents[idx2]
            alpha = np.random.uniform(0.0, 1.0)
            z_new = z_i + alpha * (z_j - z_i)
            new_samples.append(z_new)
            
        synth_latents.append(np.array(new_samples))
        synth_labels.append(np.full(n_needed, cls))
        
    balanced_latents = np.vstack(synth_latents)
    balanced_labels = np.concatenate(synth_labels)
    return balanced_latents, balanced_labels

def random_oversample(X, y_c, y_s):
    """
    Performs random oversampling ONLY on TRAIN.
    """
    unique_classes, counts = np.unique(y_c, return_counts=True)
    target_count = np.max(counts)
    
    X_res, y_c_res, y_s_res = [X], [y_c], [y_s]
    for cls in unique_classes:
        cls_mask = (y_c == cls)
        n_existing = counts[cls]
        n_needed = target_count - n_existing
        if n_needed > 0:
            indices = np.random.choice(np.where(cls_mask)[0], size=n_needed, replace=True)
            X_res.append(X[indices])
            y_c_res.append(y_c[indices])
            y_s_res.append(y_s[indices])
            
    return np.vstack(X_res), np.concatenate(y_c_res), np.concatenate(y_s_res)

def train_with_oversampling_strategy(
    train_df, val_df, test_df, feature_cols,
    strategy="latent_smote", seed=42
):
    """
    Strategy:
    A. 'none': Original training set
    B. 'random': Random oversampling on input features
    C. 'latent_smote': Extract latent embeddings using encoder -> apply latent SMOTE -> train classification head
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prep = DataPreprocessor(feature_cols=feature_cols)
    X_train, y_train_cat, y_train_score, _, _ = prepare_dataset_arrays(train_df, prep, fit_preprocessor=True)
    X_val, y_val_cat, y_val_score, _, _ = prepare_dataset_arrays(val_df, prep, fit_preprocessor=False)
    X_test, y_test_cat, y_test_score, _, _ = prepare_dataset_arrays(test_df, prep, fit_preprocessor=False)
    
    if strategy == "none":
        X_tr, y_c_tr, y_s_tr = X_train, y_train_cat, y_train_score
    elif strategy == "random":
        X_tr, y_c_tr, y_s_tr = random_oversample(X_train, y_train_cat, y_train_score)
    elif strategy == "latent_smote":
        # First train an encoder to get latent space
        from baseline import BaselineMLP
        base_model = BaselineMLP(input_dim=X_train.shape[1], hidden_dim=128, num_classes=5)
        base_model.eval()
        with torch.no_grad():
            _, _, train_lat = base_model(torch.tensor(X_train, dtype=torch.float32))
            train_lat = train_lat.numpy()
            
        lat_tr, y_c_tr = latent_space_smote(train_lat, y_train_cat)
        
        # Train classifier head on latent SMOTE representation
        classifier = nn.Linear(train_lat.shape[1], 5)
        optimizer = optim.AdamW(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        lat_ds = TensorDataset(torch.tensor(lat_tr, dtype=torch.float32), torch.tensor(y_c_tr, dtype=torch.long))
        lat_loader = DataLoader(lat_ds, batch_size=128, shuffle=True)
        
        for _ in range(50):
            for z_b, y_b in lat_loader:
                optimizer.zero_grad()
                out = classifier(z_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                
        # Evaluate using base_model + classifier
        base_model.eval()
        classifier.eval()
        with torch.no_grad():
            _, _, val_lat = base_model(torch.tensor(X_val, dtype=torch.float32))
            val_logits = classifier(val_lat)
            val_pred = torch.argmax(val_logits, dim=1).numpy()
            
            _, _, test_lat = base_model(torch.tensor(X_test, dtype=torch.float32))
            test_logits = classifier(test_lat)
            test_pred = torch.argmax(test_logits, dim=1).numpy()
            
        val_cls_metrics = evaluate_classification(y_val_cat, val_pred)
        test_cls_metrics = evaluate_classification(y_test_cat, test_pred)
        
        return {
            "strategy": strategy,
            "val_classification": val_cls_metrics,
            "test_classification": test_cls_metrics
        }
    else:
        raise ValueError(f"Unknown oversampling strategy: {strategy}")
        
    # Standard training for strategy 'none' and 'random'
    from baseline import train_baseline
    # Quick model training for comparison
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_c_tr, dtype=torch.long), torch.tensor(y_s_tr, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    from baseline import BaselineMLP
    model = BaselineMLP(input_dim=X_tr.shape[1], hidden_dim=128, num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion_c = nn.CrossEntropyLoss()
    criterion_r = nn.MSELoss()
    
    for _ in range(50):
        model.train()
        for x_b, y_c, y_s in train_loader:
            optimizer.zero_grad()
            l_c, l_r, _ = model(x_b)
            loss = criterion_c(l_c, y_c) + 0.5 * criterion_r(l_r, y_s)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_logits, _, _ = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred = torch.argmax(val_logits, dim=1).numpy()
        
        test_logits, _, _ = model(torch.tensor(X_test, dtype=torch.float32))
        test_pred = torch.argmax(test_logits, dim=1).numpy()
        
    val_cls_metrics = evaluate_classification(y_val_cat, val_pred)
    test_cls_metrics = evaluate_classification(y_test_cat, test_pred)
    
    return {
        "strategy": strategy,
        "val_classification": val_cls_metrics,
        "test_classification": test_cls_metrics
    }

if __name__ == "__main__":
    df = load_raw_dataset("d:/projects/DepDet/anon_processed_df_parquet")
    f_cols, num_cols, cat_cols, t_cols, summary = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df)
    
    results = {}
    for strat in ["none", "random", "latent_smote"]:
        res = train_with_oversampling_strategy(train_df, val_df, test_df, f_cols, strategy=strat)
        results[strat] = res
        print(f"Strategy '{strat}' -> Val Macro-F1: {res['val_classification']['macro_f1']:.4f}, Test Macro-F1: {res['test_classification']['macro_f1']:.4f}")
        
    save_metrics_to_json(results, "d:/projects/DepDet/project/results/metrics/oversampling_comparison.json")
