"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")

# Allow running as: python src/train.py or python train.py from the src/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_data, one_hot
from utils.metrics import accuracy, precision_recall_f1


def parse_arguments(args=None):
    """
    Parse command-line arguments.

    Mandatory flags :
      -d  / --dataset         'mnist' or 'fashion_mnist'
      -e  / --epochs          Number of training epochs
      -b  / --batch_size      Mini-batch size
      -l  / --loss            'cross_entropy' | 'mse'
      -o  / --optimizer       'sgd' | 'momentum' | 'nag' | 'rmsprop' | 'adam' | 'nadam'
      -lr / --learning_rate   Initial learning rate
      -wd / --weight_decay    L2 weight decay
      -nhl/ --num_layers      Number of hidden layers
      -sz / --hidden_size     Neurons per hidden layer (one value = broadcast)
      -a  / --activation      'sigmoid' | 'tanh' | 'relu'
      -w_i/ --weight_init     'random' | 'xavier'
    """
    p = argparse.ArgumentParser(
        description="Train a configurable NumPy NeuralNetwork on MNIST or Fashion-MNIST."
    )
    p.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                   choices=["mnist", "fashion_mnist"], help="Dataset.")
    p.add_argument("-e",   "--epochs",        type=int,   default=10,
                   help="Number of training epochs.")
    p.add_argument("-b",   "--batch_size",    type=int,   default=32,
                   help="Mini-batch size.")
    p.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                   choices=["cross_entropy", "mse"], help="Loss function.")
    p.add_argument("-o",   "--optimizer",     type=str,   default="adam",
                   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                   help="Optimiser.")
    p.add_argument("-lr",  "--learning_rate", type=float, default=1e-3,
                   help="Learning rate.")
    p.add_argument("-wd",  "--weight_decay",  type=float, default=0.0,
                   help="L2 weight decay.")
    p.add_argument("-nhl", "--num_layers",    type=int,   default=3,
                   help="Number of hidden layers.")
    p.add_argument("-sz",  "--hidden_size",   type=int, nargs="+", default=[128],
                   help="Hidden neurons per layer. One value = broadcast to all layers.")
    p.add_argument("-a",   "--activation",    type=str,   default="relu",
                   choices=["sigmoid", "tanh", "relu"], help="Hidden activation.")
    p.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                   choices=["random", "xavier"], help="Weight initialisation.")
    p.add_argument("--wandb_project",  type=str, default="da6401_assignment1",
                   help="W&B project name.")
    p.add_argument("--wandb_entity",   type=str, default=None,
                   help="W&B entity (team) name.")
    p.add_argument("--no_wandb",       action="store_true",
                   help="Disable Weights & Biases logging.")
    p.add_argument("--save_path",   type=str, default="models/best_model.npy",
                   help="Relative path to save model weights.")
    p.add_argument("--config_path", type=str, default="models/best_config.json",
                   help="Relative path to save model config JSON.")

    if args is None and len(sys.argv) <= 1:
        # Prevent crash in unit-test suites that import train.py without args
        args = []
    return p.parse_args(args)


def build_layer_sizes(input_dim, num_layers, hidden_sizes, output_dim):
    """Convert CLI hidden-size list into a full layer-sizes list."""
    if len(hidden_sizes) == 1:
        sizes = [hidden_sizes[0]] * num_layers
    elif len(hidden_sizes) == num_layers:
        sizes = list(hidden_sizes)
    else:
        raise ValueError(
            f"--hidden_size must have 1 value or exactly {num_layers} values, "
            f"got {len(hidden_sizes)}."
        )
    return [input_dim] + sizes + [output_dim]


def main():
    """
    Main training function.
    Trains the NeuralNetwork, logs to W&B, saves the best checkpoint by val accuracy.
    """
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------ #
    # W&B initialisation
    # ------------------------------------------------------------------ #
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
            )
        except ImportError:
            print("[WARNING] wandb not installed — running without logging.")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    print(f"Loading {args.dataset} ...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    n_classes = 10
    input_dim = X_train.shape[1]      # 784
    Y_train   = one_hot(y_train, n_classes)
    Y_val     = one_hot(y_val,   n_classes)

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    layer_sizes = build_layer_sizes(input_dim, args.num_layers,
                                    args.hidden_size, n_classes)
    activations = [args.activation] * args.num_layers

    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        weight_init=args.weight_init,
        loss=args.loss,
    )
    print(f"Architecture: {layer_sizes}  |  activation={args.activation}  "
          f"|  loss={args.loss}  |  opt={args.optimizer}")

    optimizer = get_optimizer(args.optimizer)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_val_acc = -1.0
    n_train   = X_train.shape[0]
    n_batches = int(np.ceil(n_train / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n_train)
        X_shuf, Y_shuf = X_train[perm], Y_train[perm]
        epoch_losses = []

        for bi in range(n_batches):
            s  = bi * args.batch_size
            e  = min(s + args.batch_size, n_train)
            Xb, Yb = X_shuf[s:e], Y_shuf[s:e]

            model.forward(Xb)
            epoch_losses.append(model.compute_loss(Yb))
            model.backward(Yb)
            optimizer.step(model.layers, args.learning_rate, args.weight_decay)

        train_loss = float(np.mean(epoch_losses))

        # Validation metrics
        val_preds = model.predict(X_val)
        val_acc   = accuracy(y_val, val_preds)
        model.forward(X_val)
        val_loss = model.compute_loss(Y_val)

        # Train accuracy on a random subset for speed
        sub_idx   = np.random.choice(n_train, min(5000, n_train), replace=False)
        train_acc = accuracy(y_train[sub_idx], model.predict(X_train[sub_idx]))

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if wandb_run is not None:
            import wandb  # type: ignore
            log_dict = {
                "epoch":          epoch,
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "train_accuracy": train_acc,
                "val_accuracy":   val_acc,
            }
            for li, layer in enumerate(model.layers):
                log_dict[f"grad_norm_W_layer{li + 1}"] = float(np.linalg.norm(layer.grad_W))
            wandb.log(log_dict)

        # Save best checkpoint (by val accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.save_path)
            cfg = {
                "layer_sizes":       layer_sizes,
                "activations":       activations,
                "weight_init":       args.weight_init,
                "loss":              args.loss,
                "dataset":           args.dataset,
                "optimizer":         args.optimizer,
                "learning_rate":     args.learning_rate,
                "weight_decay":      args.weight_decay,
                "batch_size":        args.batch_size,
                "epochs":            args.epochs,
                "best_val_accuracy": best_val_acc,
            }
            with open(args.config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    # ------------------------------------------------------------------ #
    # Final test evaluation using best checkpoint
    # ------------------------------------------------------------------ #
    best_model = NeuralNetwork.load(args.save_path, args.config_path)
    test_preds = best_model.predict(X_test)
    test_acc   = accuracy(y_test, test_preds)
    met        = precision_recall_f1(y_test, test_preds)

    print(f"\n=== Final Test Results (best val_acc checkpoint) ===")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  Precision     : {met['precision']:.4f}")
    print(f"  Recall        : {met['recall']:.4f}")
    print(f"  F1 Score      : {met['f1']:.4f}")

    if wandb_run is not None:
        import wandb  # type: ignore
        wandb.log({
            "test_accuracy":  test_acc,
            "test_precision": met["precision"],
            "test_recall":    met["recall"],
            "test_f1":        met["f1"],
        })
        wandb_run.finish()

    print(f"\nBest model saved : {args.save_path}")
    print(f"Config saved     : {args.config_path}")


if __name__ == "__main__":
    main()
