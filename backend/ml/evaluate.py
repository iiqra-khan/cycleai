# =============================================================================
# evaluate.py — Model evaluation and cross-validation
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def evaluate_model(model, X_test: pd.DataFrame,
                   y_test: pd.Series, model_name: str) -> dict:
    """Computes clinical-grade evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba") else y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc    = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    auprc  = average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    cm     = confusion_matrix(y_test, y_pred)

    metrics = {
        "model":            model_name,
        "accuracy":         report["accuracy"],
        "precision":        report.get("1", {}).get("precision", 0),
        "recall":           report.get("1", {}).get("recall", 0),
        "f1_score":         report.get("1", {}).get("f1-score", 0),
        "auc_roc":          auc,
        "auprc":            auprc,
        "confusion_matrix": cm.tolist()
    }

    print(f"\n  {'='*50}")
    print(f"  Model: {model_name}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return metrics


def cross_validate_model(model, X: pd.DataFrame,
                         y: pd.Series, model_name: str,
                         n_splits: int = 5) -> dict:
    """TimeSeriesSplit cross-validation — respects temporal order."""
    skf    = TimeSeriesSplit(n_splits=n_splits)
    cv_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_f1  = cross_val_score(model, X, y, cv=skf, scoring="f1",      n_jobs=-1)

    cv_results = {
        "model":       model_name,
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std":  cv_auc.std(),
        "cv_f1_mean":  cv_f1.mean(),
        "cv_f1_std":   cv_f1.std(),
    }

    print(f"  [CV] {model_name} "
          f"(on training split only — 80% of participant history)")
    print(f"  AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  F1:      {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

    return cv_results