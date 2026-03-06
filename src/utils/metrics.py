"""
Evaluation Metrics
Accuracy, Precision, Recall, F1, Confusion Matrix (via scikit-learn)
"""
import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro-averaged precision, recall, and F1 score."""
    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred,    average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred,        average="macro", zero_division=0)),
    }


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10
) -> np.ndarray:
    """Return (n_classes × n_classes) confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
