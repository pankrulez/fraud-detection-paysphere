from typing import Dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score


def compute_classification_metrics(y_true, y_prob, threshold: float) -> Dict[str, float]:
    """
    Compute fraud-specific metrics: precision, recall at given threshold, ROC AUC, PR AUC.
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    return metrics