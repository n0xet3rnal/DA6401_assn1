"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-z))."""
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid w.r.t. its pre-activation z."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh":    (tanh,    tanh_derivative),
    "relu":    (relu,    relu_derivative),
}


def get_activation(name: str):
    """Return (activation_fn, derivative_fn) for the given activation name."""
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]