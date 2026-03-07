"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Cross-Entropy
# --------------------------------------------------------------------------- #

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Cross-entropy loss. y_pred are probabilities (after softmax)."""
    batch_size = y_pred.shape[0]
    log_probs = np.log(np.clip(y_pred, 1e-12, 1.0))
    return -np.sum(y_true * log_probs) / batch_size


def cross_entropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of CE+Softmax w.r.t. logits.
    Combined CE+Softmax gradient simplifies to (y_pred − y_true) / batch_size.
    """
    batch_size = y_pred.shape[0]
    return (y_pred - y_true) / batch_size


# --------------------------------------------------------------------------- #
# Mean Squared Error
# --------------------------------------------------------------------------- #

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss (averaged over batch)."""
    batch_size = y_pred.shape[0]
    return np.sum((y_pred - y_true) ** 2) / batch_size


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. y_pred (probabilities, before Softmax Jacobian)."""
    batch_size = y_pred.shape[0]
    return 2.0 * (y_pred - y_true) / batch_size


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mse":           (mse_loss,           mse_grad),
}


def get_loss(name: str):
    """Return (loss_fn, grad_fn) for the given loss name."""
    name = name.lower()
    if name not in _LOSSES:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: {list(_LOSSES)}"
        )
    return _LOSSES[name]