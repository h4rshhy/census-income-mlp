"""Threshold selection and classification metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def best_threshold(y_val, p_val):
    """Pick the threshold on the validation set that maximises F1."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (p_val >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metrics(y_true, p, thr):
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, p),
            "pr_auc": average_precision_score(y_true, p),
            "cm": confusion_matrix(y_true, pred),
            "threshold": thr}
