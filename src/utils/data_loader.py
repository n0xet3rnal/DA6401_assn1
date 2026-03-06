"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets via keras.datasets
"""
import numpy as np


_DATASET_NAMES = {"mnist", "fashion_mnist"}


def load_data(dataset_name: str):
    """
    Load, preprocess, and split into train / val / test.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
        X arrays are float64, shape (N, 784), normalised to [0, 1].
        y arrays are integer labels, shape (N,).
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {_DATASET_NAMES}"
        )

    # Try keras, fall back to tensorflow.keras
    try:
        if dataset_name == "mnist":
            from keras.datasets import mnist as ds          # type: ignore
        else:
            from keras.datasets import fashion_mnist as ds  # type: ignore
    except ModuleNotFoundError:
        try:
            if dataset_name == "mnist":
                from tensorflow.keras.datasets import mnist as ds          # type: ignore
            else:
                from tensorflow.keras.datasets import fashion_mnist as ds  # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Could not import keras or tensorflow.keras. "
                "Install with: pip install tensorflow-cpu  or  pip install keras tensorflow"
            )

    (X_raw_train, y_train_all), (X_raw_test, y_test) = ds.load_data()

    X_all  = preprocess(X_raw_train)
    X_test = preprocess(X_raw_test)

    # 90/10 train/val split
    n_total = X_all.shape[0]
    n_val   = int(0.1 * n_total)
    idx     = np.random.permutation(n_total)

    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X_all[train_idx], y_train_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_train_all[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess(X: np.ndarray) -> np.ndarray:
    """Flatten (28×28 → 784) and normalise pixel values to [0, 1]."""
    return X.reshape(X.shape[0], -1).astype(np.float64) / 255.0


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Convert integer label vector to one-hot matrix (N, n_classes)."""
    matrix = np.zeros((len(y), n_classes), dtype=np.float64)
    matrix[np.arange(len(y)), y] = 1.0
    return matrix
