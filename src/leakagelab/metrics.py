"""Metrics shared by solver validation and author grader; contains no labels."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss, precision_score, recall_score)


def metrics(y, probabilities, threshold):
    p = np.asarray(probabilities, dtype=float)
    pred = p >= threshold
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "threshold": float(threshold), "n": int(len(y)), "prevalence": float(np.mean(y))}
