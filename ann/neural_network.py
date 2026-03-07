"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import json
import numpy as np
try:
    from ann.neural_layer import DenseLayer
    from ann.activations import softmax
    from ann.objective_functions import get_loss
except ImportError:
    # Fallback for direct file execution by autograders
    from neural_layer import DenseLayer
    from activations import softmax
    from objective_functions import get_loss


class NeuralNetwork:
    """
    Configurable NeuralNetwork built exclusively with NumPy.

    Accepts either a parsed CLI args namespace (from train.py) or explicit
    keyword arguments when instantiated directly.

       -------------------------
    * forward(X)         — returns softmax probabilities, shape (b, n_classes)
    * backward(y_true, y_pred)
                         — computes gradients; stores self.grad_W and self.grad_b
                           as object arrays where index 0 = **last** (output) layer
    * self.grad_W[i]     — weight gradient for the i-th layer from the output
    * self.grad_b[i]     — bias gradient   for the i-th layer from the output
    * layer.grad_W / layer.grad_b — also exposed on every DenseLayer object
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, cli_args=None, **kwargs):
        """
        Build the network from cli_args (argparse.Namespace) or keyword args.

        Keyword args (used when not passing cli_args):
            layer_sizes  : list of ints [input, h1, h2, ..., output]
            activations  : list of str or single str for hidden layers
            weight_init  : 'random' | 'xavier'
            loss         : 'cross_entropy' | 'mse'
        """
        if cli_args is not None and hasattr(cli_args, "num_layers"):
            # Build from argparse namespace produced by train.py
            n_hidden    = cli_args.num_layers
            hidden_size = cli_args.hidden_size   # list
            activation  = cli_args.activation    # single str
            weight_init = cli_args.weight_init
            loss        = cli_args.loss
            input_dim   = getattr(cli_args, "input_dim", 784)
            n_classes   = getattr(cli_args, "n_classes",  10)

            if len(hidden_size) == 1:
                sizes = [hidden_size[0]] * n_hidden
            elif len(hidden_size) == n_hidden:
                sizes = list(hidden_size)
            else:
                raise ValueError(
                    f"--hidden_size must have 1 value or exactly {n_hidden}, "
                    f"got {len(hidden_size)}."
                )

            layer_sizes = [input_dim] + sizes + [n_classes]
            activations = [activation] * n_hidden
        else:
            # Direct construction (used by verify.py / tests)
            layer_sizes = kwargs.get("layer_sizes")
            activations = kwargs.get("activations", None)
            weight_init = kwargs.get("weight_init", "xavier")
            loss        = kwargs.get("loss", "cross_entropy")

        n_hidden = len(layer_sizes) - 2

        # Normalise activations
        if activations is None:
            activations = ["relu"] * n_hidden
        elif isinstance(activations, str):
            activations = [activations] * n_hidden
        elif len(activations) != n_hidden:
            raise ValueError(
                f"Expected {n_hidden} activations, got {len(activations)}."
            )

        self.layer_sizes      = layer_sizes
        self.activation_names = activations
        self.weight_init      = weight_init
        self.loss_name        = loss

        self.loss_fn, self.loss_grad_fn = get_loss(loss)

        # Build layers: n_hidden hidden layers + 1 linear output layer
        self.layers: list = []
        for i in range(n_hidden):
            self.layers.append(DenseLayer(
                in_dim=layer_sizes[i],
                out_dim=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init,
            ))
        self.layers.append(DenseLayer(
            in_dim=layer_sizes[-2],
            out_dim=layer_sizes[-1],
            activation="linear",
            weight_init=weight_init,
        ))

        self.probs: np.ndarray = None

        # Grader-required gradient arrays (populated by backward())
        self.grad_W = None
        self.grad_b = None

    # ------------------------------------------------------------------ #
    # Forward & Backward
    # ------------------------------------------------------------------ #

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.
        Returns logits passed through softmax — shape (b, n_classes).
        X is shape (b, D_in).
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.probs = softmax(out)
        return self.probs

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> tuple:
        """
        Backward propagation to compute gradients.

        For CE+Softmax the combined gradient simplifies to (probs - y_true)/b.
        For MSE we apply the Softmax Jacobian before entering the layer chain.

        Stores results in:
            self.grad_W  — object array, index 0 = last (output) layer
            self.grad_b  — object array, index 0 = last (output) layer

        Returns (self.grad_W, self.grad_b).
        """
        # Gradient of loss w.r.t. softmax output probabilities
        grad_probs = self.loss_grad_fn(self.probs, y_true)

        # For CE the CE+Softmax combined gradient is already w.r.t. logits.
        # For MSE we need to propagate through the Softmax Jacobian.
        if self.loss_name == "cross_entropy":
            grad_logits = grad_probs
        else:
            p   = self.probs
            dot = np.sum(grad_probs * p, axis=1, keepdims=True)
            grad_logits = p * (grad_probs - dot)

        # Backprop through layers in reverse; accumulate in reversed order
        grad_W_list = []
        grad_b_list = []

        grad = grad_logits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Store as object arrays — index 0 = last (output) layer
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def compute_loss(self, y_true: np.ndarray) -> float:
        """Compute scalar loss using the current forward-pass probabilities."""
        return self.loss_fn(self.probs, y_true)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions."""
        return np.argmax(self.forward(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probability matrix."""
        return self.forward(X)

    def update_weights(self):
        """Placeholder — weight updates are handled externally by optimizers."""
        pass

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """Minimal batch-training loop (use train.py for full CLI training)."""
        from .optimizers import get_optimizer
        optimizer = get_optimizer("adam")
        n = X_train.shape[0]
        for _ in range(epochs):
            perm = np.random.permutation(n)
            X_s, Y_s = X_train[perm], y_train[perm]
            for s in range(0, n, batch_size):
                Xb = X_s[s: s + batch_size]
                Yb = Y_s[s: s + batch_size]
                self.forward(Xb)
                self.backward(Yb)
                optimizer.step(self.layers, lr=1e-3)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return accuracy on the given data split."""
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        return {"accuracy": acc}

    # ------------------------------------------------------------------ #
    # Weight persistence
    # ------------------------------------------------------------------ #

    def get_weights(self) -> dict:
        """Return a dict mapping 'Wi' / 'bi' to numpy arrays."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict) -> None:
        """Load weights from a dict mapping 'Wi' / 'bi' to numpy arrays."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def save(self, path: str) -> None:
        """Serialise all layer weights/biases to a .npy file."""
        params = [{"W": layer.W, "b": layer.b} for layer in self.layers]
        np.save(path, params, allow_pickle=True)

    @classmethod
    def load(cls, weights_path: str, config_path: str) -> "NeuralNetwork":
        """Load a saved model from .npy weights and a JSON config file."""
        with open(config_path, "r") as f:
            cfg = json.load(f)

        model = cls(
            layer_sizes=cfg["layer_sizes"],
            activations=cfg["activations"],
            weight_init=cfg["weight_init"],
            loss=cfg["loss"],
        )
        params = np.load(weights_path, allow_pickle=True)
        for layer, p in zip(model.layers, params):
            layer.W = p["W"]
            layer.b = p["b"]
        return model

    # ------------------------------------------------------------------ #
    # Gradient verification
    # ------------------------------------------------------------------ #

    def gradient_check(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        eps: float = 1e-5,
        n_checks: int = 20,
    ) -> float:
        """
        Compare analytical (backward()) gradients vs numerical finite-difference
        gradients on the first layer's weights.
        Returns the relative error: |analytic − numeric| / (|analytic| + |numeric|).
        """
        self.forward(X)
        self.backward(y_true)
        layer = self.layers[0]
        W_flat   = layer.W.flatten()
        grad_flat = layer.grad_W.flatten()

        rng     = np.random.default_rng(0)
        indices = rng.choice(len(W_flat), size=min(n_checks, len(W_flat)), replace=False)

        analytic = grad_flat[indices]
        numeric  = np.zeros(len(indices))

        for k, idx in enumerate(indices):
            i, j = np.unravel_index(idx, layer.W.shape)
            layer.W[i, j] += eps
            lp = self.loss_fn(self.forward(X), y_true)
            layer.W[i, j] -= 2 * eps
            lm = self.loss_fn(self.forward(X), y_true)
            layer.W[i, j] += eps          # restore
            numeric[k] = (lp - lm) / (2 * eps)

        numerator   = np.linalg.norm(analytic - numeric)
        denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
        return float(numerator / (denominator + 1e-12))

    def __repr__(self):
        layers_str = " → ".join(repr(l) for l in self.layers)
        return f"NeuralNetwork([{layers_str}], loss={self.loss_name})"
