"""
Visualisation Utilities
Confusion matrix plot and W&B sample image table
"""
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
):
    """
    Plot a labelled confusion matrix and optionally save it to disk.

    Parameters
    ----------
    cm         : (n, n) confusion matrix from compute_confusion_matrix()
    class_names: list of class label strings
    title      : plot title
    save_path  : if given, save the figure to this path (PNG, 150 dpi)
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping confusion matrix plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def log_sample_images_wandb(
    X, y, class_names, wandb_run, n_per_class: int = 5, table_key: str = "sample_images"
):
    """Log a table of sample images (one per class) to Weights & Biases."""
    import wandb  # type: ignore

    table = wandb.Table(columns=["class_id", "class_name", "sample_index", "image"])
    for cid, cname in enumerate(class_names):
        idxs = np.where(y == cid)[0][:n_per_class]
        for sample_i, idx in enumerate(idxs):
            img = (X[idx].reshape(28, 28) * 255).astype(np.uint8)
            table.add_data(cid, cname, sample_i + 1, wandb.Image(img))
    wandb_run.log({table_key: table})
