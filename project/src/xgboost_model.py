import numpy as np
import pandas as pd
import os
import sys
import json
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluation import evaluate_classification, save_metrics_to_json, plot_confusion_matrix

def compute_multiclass_sample_weights(y_train: np.ndarray, num_classes: int | list = 5) -> np.ndarray:
    """
    Computes per-sample weights strictly from training labels to address class imbalance:
        w_c = N / (C * N_c)
        sample_weight_i = w_{y_i}
    """
    if hasattr(num_classes, '__len__'):
        C = len(num_classes)
    else:
        C = int(num_classes)
        
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_weight_map = {}
    for c, count in zip(classes, counts):
        class_weight_map[c] = total_samples / (C * count)
        
    sample_weights = np.array([class_weight_map[int(y)] for y in y_train], dtype=np.float32)
    return sample_weights

def build_xgboost_classifier(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: float = 3.0,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    seed: int = 42,
    random_state: int = None
) -> xgb.XGBClassifier:
    """
    Builds an XGBClassifier with multiclass probability objective.
    """
    actual_seed = random_state if random_state is not None else seed
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=actual_seed,
        tree_method="hist",
        early_stopping_rounds=15
    )

def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    use_sample_weights: bool = False,
    params: dict | None = None,
    seed: int = 42,
    checkpoint_name: str = "xgboost_best.json"
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Trains XGBoost using early stopping on validation loss, evaluates on validation data,
    and returns model and metrics.
    """
    if params is None:
        params = {}
        
    cfg = {
        "n_estimators": params.get("n_estimators", 200),
        "max_depth": params.get("max_depth", 5),
        "learning_rate": params.get("learning_rate", 0.05),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_weight": params.get("min_child_weight", 3.0),
        "reg_alpha": params.get("reg_alpha", 0.1),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "seed": seed
    }
    
    model = build_xgboost_classifier(**cfg)
    
    sample_weight = None
    if use_sample_weights:
        sample_weight = compute_multiclass_sample_weights(y_train, num_classes=5)
        
    # Fit model with validation set monitoring
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # Save checkpoint
    ckpt_dir = "d:/projects/DepDet/project/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
    model.save_model(ckpt_path)
    
    # Validation predictions and probabilities
    val_probs = model.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = evaluate_classification(y_val, val_preds, y_prob=val_probs)
    
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_probs = model.predict_proba(X_test)
        test_preds = np.argmax(test_probs, axis=1)
        test_metrics = evaluate_classification(y_test, test_preds, y_prob=test_probs)
        
    results = {
        "config": cfg,
        "use_sample_weights": use_sample_weights,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") and model.best_iteration is not None else cfg["n_estimators"],
        "val_classification": val_metrics,
        "test_classification": test_metrics
    }
    
    return model, results
