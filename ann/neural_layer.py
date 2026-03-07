"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import get_activation


# --------------------------------------------------------------------------- #
# Weight Initialisation Functions
# --------------------------------------------------------------------------- #

def _random_init(shape: tuple) -> np.ndarray:
    """Small random weights: N(0, 0.01)."""
    return np.random.randn(*shape) * 0.01


def _xavier_init(shape: tuple) -> np.ndarray:
    """Xavier / Glorot uniform initialisation."""
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


_INITS = {
    "random": _random_init,
    "xavier": _xavier_init,
}


def get_init(name: str):
    """Return the weight initialisation function for *name*."""
    name = name.lower()
    if name not in _INITS:
        raise ValueError(
            f"Unknown weight init '{name}'. Choose from: {list(_INITS)}"
        )
    return _INITS[name]


# --------------------------------------------------------------------------- #
# DenseLayer
# --------------------------------------------------------------------------- #

class DenseLayer:
    """
    Single fully-connected (dense) layer.

    After forward():
        self.input   — input activations  (b, in_dim)
        self.z       — pre-activation     (b, out_dim)
        self.output  — post-activation    (b, out_dim)

    After backward():
        self.grad_W  — weight gradient    (in_dim, out_dim)  [required by grader]
        self.grad_b  — bias gradient      (1, out_dim)       [required by grader]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        weight_init: str = "xavier",
    ):
        init_fn = get_init(weight_init)
        self.W = init_fn((in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.activation_name  = activation
        self.weight_init_name = weight_init

        # Support a passthrough "linear" activation for the output layer
        if activation == "linear":
            self._act       = lambda z: z
            self._act_deriv = lambda z: np.ones_like(z)
        else:
            self._act, self._act_deriv = get_activation(activation)

        # Initialise gradient buffers (exposed to grader)
        self.grad_W: np.ndarray = np.zeros_like(self.W)
        self.grad_b: np.ndarray = np.zeros_like(self.b)

        # Cache for backprop
        self.input:  np.ndarray = None
        self.z:      np.ndarray = None
        self.output: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute z = X·W + b, then apply activation. Returns output."""
        self.input  = X
        self.z      = X @ self.W + self.b
        self.output = self._act(self.z)
        return self.output

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagate *grad_out* (gradient w.r.t. this layer's output).
        Stores self.grad_W, self.grad_b; returns gradient w.r.t. input.
        """
        delta = grad_out * self._act_deriv(self.z)   # (b, out_dim)
        self.grad_W = self.input.T @ delta            # (in_dim, out_dim)
        self.grad_b = np.sum(delta, axis=0, keepdims=True)  # (1, out_dim)
        return delta @ self.W.T                       # (b, in_dim)

    def get_params(self):
        return self.W, self.b

    def set_params(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b

    def __repr__(self):
        return (
            f"DenseLayer(in={self.in_dim}, out={self.out_dim}, "
            f"act={self.activation_name}, init={self.weight_init_name})"
        )