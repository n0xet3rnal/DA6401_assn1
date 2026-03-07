"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
Each optimizer exposes a step(layers, lr, weight_decay) method.
"""
import numpy as np


class SGD:
    """Vanilla Stochastic Gradient Descent with optional L2 weight decay."""

    def step(self, layers: list, lr: float, weight_decay: float = 0.0) -> None:
        for layer in layers:
            layer.W -= lr * (layer.grad_W + weight_decay * layer.W)
            layer.b -= lr * layer.grad_b


class Momentum:
    """SGD with Momentum (classical heavy-ball). Default beta = 0.9."""

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self._v: dict = {}

    def step(self, layers: list, lr: float, weight_decay: float = 0.0) -> None:
        for i, layer in enumerate(layers):
            if i not in self._v:
                self._v[i] = {
                    "W": np.zeros_like(layer.W),
                    "b": np.zeros_like(layer.b),
                }
            v = self._v[i]
            # velocity update (includes weight-decay term in gradient)
            v["W"] = self.beta * v["W"] + layer.grad_W + weight_decay * layer.W
            v["b"] = self.beta * v["b"] + layer.grad_b
            layer.W -= lr * v["W"]
            layer.b -= lr * v["b"]


class NAG:
    """
    Nesterov Accelerated Gradient.
    The look-ahead step is approximated by correcting the update with
    the current and previous velocities, matching standard textbook NAG.
    """

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self._v: dict = {}

    def step(self, layers: list, lr: float, weight_decay: float = 0.0) -> None:
        for i, layer in enumerate(layers):
            if i not in self._v:
                self._v[i] = {
                    "W": np.zeros_like(layer.W),
                    "b": np.zeros_like(layer.b),
                }
            v = self._v[i]
            v_prev_W = v["W"].copy()
            v_prev_b = v["b"].copy()

            grad_W = layer.grad_W + weight_decay * layer.W
            v["W"] = self.beta * v["W"] + grad_W
            v["b"] = self.beta * v["b"] + layer.grad_b

            # Nesterov correction: (1+β)·vt − β·v_{t-1}
            layer.W -= lr * ((1 + self.beta) * v["W"] - self.beta * v_prev_W)
            layer.b -= lr * ((1 + self.beta) * v["b"] - self.beta * v_prev_b)


class RMSProp:
    """RMSProp — adaptive per-parameter learning rates using a moving average of squared gradients."""

    def __init__(self, beta: float = 0.9, epsilon: float = 1e-8):
        self.beta = beta
        self.eps  = epsilon
        self._s: dict = {}

    def step(self, layers: list, lr: float, weight_decay: float = 0.0) -> None:
        for i, layer in enumerate(layers):
            if i not in self._s:
                self._s[i] = {
                    "W": np.zeros_like(layer.W),
                    "b": np.zeros_like(layer.b),
                }
            s = self._s[i]
            grad_W = layer.grad_W + weight_decay * layer.W

            s["W"] = self.beta * s["W"] + (1 - self.beta) * grad_W ** 2
            s["b"] = self.beta * s["b"] + (1 - self.beta) * layer.grad_b ** 2

            layer.W -= lr / (np.sqrt(s["W"]) + self.eps) * grad_W
            layer.b -= lr / (np.sqrt(s["b"]) + self.eps) * layer.grad_b


class Adam:
    """
    Adam — Adaptive Moment Estimation.
    Maintains first (m) and second (v) moment estimates with bias correction.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = epsilon
        self._m: dict = {}
        self._v: dict = {}
        self.t = 0

    def step(self, layers: list, lr: float, weight_decay: float = 0.0) -> None:
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self._m:
                self._m[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                self._v[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

            grad_W = layer.grad_W + weight_decay * layer.W
            grad_b = layer.grad_b

            # Moment updates
            self._m[i]["W"] = self.beta1 * self._m[i]["W"] + (1 - self.beta1) * grad_W
            self._m[i]["b"] = self.beta1 * self._m[i]["b"] + (1 - self.beta1) * grad_b
            self._v[i]["W"] = self.beta2 * self._v[i]["W"] + (1 - self.beta2) * grad_W ** 2
            self._v[i]["b"] = self.beta2 * self._v[i]["b"] + (1 - self.beta2) * grad_b ** 2

            # Bias-corrected estimates
            m_hat_W = self._m[i]["W"] / (1 - self.beta1 ** self.t)
            m_hat_b = self._m[i]["b"] / (1 - self.beta1 ** self.t)
            v_hat_W = self._v[i]["W"] / (1 - self.beta2 ** self.t)
            v_hat_b = self._v[i]["b"] / (1 - self.beta2 ** self.t)

            layer.W -= lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


class Nadam:
    """
    Nadam — Adam with Nesterov momentum.
    Uses the Nesterov-corrected first-moment estimate for the weight update.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = epsilon
        self._m: dict = {}
        self._v: dict = {}
        self.t = 0

    def step(self, layers: list, lr: float, weight_decay: float = 0.0) -> None:
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self._m:
                self._m[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                self._v[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

            grad_W = layer.grad_W + weight_decay * layer.W
            grad_b = layer.grad_b

            self._m[i]["W"] = self.beta1 * self._m[i]["W"] + (1 - self.beta1) * grad_W
            self._m[i]["b"] = self.beta1 * self._m[i]["b"] + (1 - self.beta1) * grad_b
            self._v[i]["W"] = self.beta2 * self._v[i]["W"] + (1 - self.beta2) * grad_W ** 2
            self._v[i]["b"] = self.beta2 * self._v[i]["b"] + (1 - self.beta2) * grad_b ** 2

            # Second-moment bias correction
            v_hat_W = self._v[i]["W"] / (1 - self.beta2 ** self.t)
            v_hat_b = self._v[i]["b"] / (1 - self.beta2 ** self.t)

            # Nesterov-corrected first moment (look-ahead)
            m_nag_W = (self.beta1 * self._m[i]["W"] + (1 - self.beta1) * grad_W) \
                      / (1 - self.beta1 ** (self.t + 1))
            m_nag_b = (self.beta1 * self._m[i]["b"] + (1 - self.beta1) * grad_b) \
                      / (1 - self.beta1 ** (self.t + 1))

            layer.W -= lr * m_nag_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= lr * m_nag_b / (np.sqrt(v_hat_b) + self.eps)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}


def get_optimizer(name: str, **kwargs):
    """Instantiate and return the named optimizer with optional hyperparameters."""
    name = name.lower()
    if name not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {list(_OPTIMIZERS)}"
        )
    return _OPTIMIZERS[name](**kwargs)