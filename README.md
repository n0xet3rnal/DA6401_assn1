# DA6401 Assignment 1 – NumPy MLP

A fully configurable, modular **Multi-Layer Perceptron** built with **NumPy only** for image classification on MNIST and Fashion-MNIST.

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
│   │   ├── data_loader.py        # load_data, preprocess, one_hot
│   │   ├── metrics.py            # accuracy, precision, recall, F1
│   │   └── visualize.py          # confusion matrix, W&B image table
│   ├── train.py                  # CLI training script
│   └── inference.py              # CLI inference + metrics script
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
# From project root
python src/train.py -d mnist -e 10 -b 32 -l cross_entropy \
                   -o adam -lr 0.001 -wd 0.0 \
                   -nhl 3 -sz 128 -a relu -w_i xavier
```

### 3. Inference
```bash
python src/inference.py --model models/best_model.npy \
                        --config models/best_config.json \
                        -d mnist --plot_cm
```

## CLI Arguments

| Flag | Long | Description | Default |
|------|------|-------------|---------|
| `-d` | `--dataset` | `mnist` or `fashion_mnist` | `mnist` |
| `-e` | `--epochs` | Training epochs | `10` |
| `-b` | `--batch_size` | Mini-batch size | `32` |
| `-l` | `--loss` | `cross_entropy` or `mse` | `cross_entropy` |
| `-o` | `--optimizer` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | `adam` |
| `-lr` | `--learning_rate` | Learning rate | `0.001` |
| `-wd` | `--weight_decay` | L2 regularisation | `0.0` |
| `-nhl` | `--num_layers` | Number of hidden layers | `3` |
| `-sz` | `--hidden_size` | Neurons per hidden layer | `128` |
| `-a` | `--activation` | `sigmoid`, `tanh`, `relu` | `relu` |
| `-w_i` | `--weight_init` | `random`, `xavier` | `xavier` |

## Outputs

After training, you will find:
- `models/best_model.npy` — serialised weights (best val accuracy)
- `models/best_config.json` — matching hyperparameter config

## WandB Report
[Report Link](https://wandb.ai/be22b022-indian-institute-of-technology-madras/da6401_assignment1/reports/DA6401-Assignment-1-BE22B022--VmlldzoxNjA1MTgwNg?accessToken=toja56uklr2mc9mpz97ehsxeqb422384631ril7cacdajnz0qy36lezc6d5znc46)

