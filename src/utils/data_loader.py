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

    import os
    import gzip
    import urllib.request
    
    def download_and_extract(url, filepath):
        if not os.path.exists(filepath):
            print(f"Downloading {url} to {filepath}...")
            urllib.request.urlretrieve(url, filepath)
        with gzip.open(filepath, 'rb') as f:
            return f.read()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == "mnist":
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    else:
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    urls = [base_url + f for f in files]
    filepaths = [os.path.join(data_dir, f) for f in files]

    train_images_raw = download_and_extract(urls[0], filepaths[0])
    train_labels_raw = download_and_extract(urls[1], filepaths[1])
    test_images_raw = download_and_extract(urls[2], filepaths[2])
    test_labels_raw = download_and_extract(urls[3], filepaths[3])

    # Parse idx files
    X_raw_train = np.frombuffer(train_images_raw, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    y_train_all = np.frombuffer(train_labels_raw, dtype=np.uint8, offset=8)
    X_raw_test = np.frombuffer(test_images_raw, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    y_test = np.frombuffer(test_labels_raw, dtype=np.uint8, offset=8)

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
