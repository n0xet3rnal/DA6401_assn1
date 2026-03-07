# DA6401 Assignment 1 – NumPy MLP

A fully configurable, modular **Multi-Layer Perceptron (MLP)** built entirely from scratch using **only NumPy**. This project is designed for image classification tasks on the **MNIST** and **Fashion-MNIST** datasets. It features a highly modular object-oriented design, allowing you to easily swap out loss functions, optimizers, activation functions, and network architectures.

## Project Structure

```
├── models/                 # Saved weights & configs
├── notebooks/              # W&B demo notebook
├── sweep_config.yaml       # W&B Sweep configuration
├── ann/
│   ├── activations.py        # sigmoid, tanh, relu, softmax
│   ├── neural_layer.py       # DenseLayer (forward/backward, grad_W, grad_b)
│   ├── neural_network.py     # NeuralNetwork class (forward/backward/predict/save/load)
│   ├── objective_functions.py# cross_entropy & mse losses + gradients
│   └── optimizers.py         # SGD, Momentum, NAG, RMSProp, Adam, Nadam
├── src/
│   ├── utils/
│   │   ├── data_loader.py        # Dataset fetching, preprocessing, and one-hot encoding
│   │   ├── metrics.py            # Accuracy, Precision, Recall, F1-Score
│   │   └── visualize.py          # Plotting utilities and confusion matrices
│   ├── train.py                  # CLI training script
│   └── inference.py              # CLI evaluation and inference script
├── requirements.txt        # Required Python packages
└── README.md
```

## Quick Start

### 1. Install Dependencies
Make sure you have Python installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Train a Model
Run the `train.py` script from the project root. The script will automatically download the dataset (if not present), train the model, log metrics to W&B, and save the best checkpoint.

```bash
# Example: Train a 3-layer MLP on MNIST with Adam, ReLU, and Cross-Entropy
python src/train.py -d mnist -e 10 -b 32 -l cross_entropy \
                   -o adam -lr 0.001 -wd 0.0 \
                   -nhl 3 -sz 128 -a relu -w_i xavier
```

### 3. Run Inference
Evaluate your saved model on the test set and optionally generate a confusion matrix.
```bash
python src/inference.py --model src/best_model.npy \
                        --config src/best_config.json \
                        --dataset mnist --plot_cm
```

## 🛠 Command-Line Interface (CLI) Arguments

The `train.py` script accepts the following parameters to fully customize your model:

| Flag | Long Argument | Description | Default |
|------|---------------|-------------|---------|
| `-d` | `--dataset` | Choose between `mnist` or `fashion_mnist` | `mnist` |
| `-e` | `--epochs` | Number of training epochs | `10` |
| `-b` | `--batch_size` | Mini-batch size for training | `32` |
| `-l` | `--loss` | Loss function: `cross_entropy` or `mse` | `cross_entropy` |
| `-o` | `--optimizer` | Optimization algorithm: `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | `adam` |
| `-lr` | `--learning_rate` | Initial learning rate | `0.001` |
| `-wd` | `--weight_decay` | L2 weight decay (regularization) factor | `0.0` |
| `-nhl` | `--num_layers` | Number of hidden layers in the network | `3` |
| `-sz` | `--hidden_size` | Number of neurons per hidden layer | `128` |
| `-a` | `--activation` | Hidden layer activation: `sigmoid`, `tanh`, `relu` | `relu` |
| `-w_i` | `--weight_init` | Weight initialization strategy: `random`, `xavier` | `xavier` |

*Note: W&B logging can be disabled by passing the `--no_wandb` flag.*

##  Hyperparameter Sweeps (Weights & Biases)

This project supports **Bayesian Hyperparameter Sweeps** via W&B to automatically find the best network configuration.

1. **Initialize the Sweep:**
```bash
wandb sweep sweep_config.yaml
```
2. **Start the Agent:** W&B will output a sweep ID. Run the following command to start searching:
```bash
wandb agent [USERNAME]/[PROJECT]/[SWEEP_ID]
```

##  Outputs & Checkpoints

By default, standard training generates the following files in the `src/` directory (or wherever specified via `--save_path`):
- `best_model.npy`: Serialized NumPy arrays containing the network weights and biases that achieved the highest validation accuracy.
- `best_config.json`: The hyperparameter configuration used to train the best model.

##  W&B Report Summary
[View Full W&B Project Report](https://wandb.ai/be22b022-indian-institute-of-technology-madras/da6401_assignment1/reports/DA6401-Assignment-1-BE22B022--VmlldzoxNjA1MTgwNg?accessToken=toja56uklr2mc9mpz97ehsxeqb422384631ril6cacdajnz0qy36lezc6d5znc46)
