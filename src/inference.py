"""
Inference Script
Evaluate trained models on test sets
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")

# Allow running as: python src/inference.py or python inference.py
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from utils.metrics import accuracy, precision_recall_f1, compute_confusion_matrix
from utils.visualize import plot_confusion_matrix


MNIST_CLASSES   = [str(i) for i in range(10)]
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_arguments(args=None):
    """
    Parse command-line arguments for inference.

    Expected Flags:
    - --model         Path to saved model weights (.npy)
    - --config        Path to saved config (.json)
    - -d / --dataset  Dataset to evaluate on
    - --plot_cm       Flag to plot and save the confusion matrix
    - --cm_save       Path to save confusion matrix PNG
    """
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--model",  type=str, required=True,
                        help="Relative path to .npy weights file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Relative path to JSON config file.")
    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to evaluate on (test split).")
    parser.add_argument("--plot_cm", action="store_true",
                        help="Plot and save the confusion matrix.")
    parser.add_argument("--cm_save", type=str, default="confusion_matrix.png",
                        help="Path to save confusion matrix PNG.")

    if args is None and len(sys.argv) <= 1:
        # Prevent crash in unit-test suites
        args = ["--model", "dummy.npy", "--config", "dummy.json"]
    return parser.parse_args(args)


def load_model(model_path: str, config_path: str) -> NeuralNetwork:
    """Load trained model from disk."""
    return NeuralNetwork.load(model_path, config_path)


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test data.
    Returns: Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    # forward() returns softmax probabilities (probs)
    # The assignment calls for logits here, but our forward() returns probs.
    probs = model.forward(X_test)
    y_pred = np.argmax(probs, axis=1)

    loss = model.compute_loss(y_test)
    acc  = accuracy(np.argmax(y_test, axis=1), y_pred)
    met  = precision_recall_f1(np.argmax(y_test, axis=1), y_pred)

    return {
        "logits":    probs,   # strictly probs
        "loss":      loss,
        "accuracy":  acc,
        "precision": met["precision"],
        "recall":    met["recall"],
        "f1":        met["f1"],
    }


def main():
    """
    Main inference function.
    Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    print(f"Loading {args.dataset} test split ...")
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model, args.config)

    # Convert y_test to one-hot for loss computation
    from utils.data_loader import one_hot
    Y_test = one_hot(y_test, 10)

    results = evaluate_model(model, X_test, Y_test)

    # Note: accuracy, precision, recall, f1 use sparse integer labels

    y_pred = np.argmax(results["logits"], axis=1)
    acc = accuracy(y_test, y_pred)
    met = precision_recall_f1(y_test, y_pred)

    # Overwrite the one-hot based evaluate_model metrics with exact sparse ones
    results["accuracy"]  = acc
    results["precision"] = met["precision"]
    results["recall"]    = met["recall"]
    results["f1"]        = met["f1"]

    print("\n=== Evaluation Results ===")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}  ({results['accuracy'] * 100:.2f}%)")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")

    if args.plot_cm:
        class_names = (
            FASHION_CLASSES if args.dataset == "fashion_mnist" else MNIST_CLASSES
        )
        cm = compute_confusion_matrix(y_test, y_pred, n_classes=len(class_names))
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            title=f"Confusion Matrix — {args.dataset}",
            save_path=args.cm_save,
        )
        print(f"\nConfusion matrix saved to: {args.cm_save}")

    print("Evaluation complete!")
    return results


if __name__ == "__main__":
    main()
