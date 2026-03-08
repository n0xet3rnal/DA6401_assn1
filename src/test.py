import os
import argparse
import numpy as np
from ann.neural_network import NeuralNetwork

src_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(src_dir, 'best_model.npy')
config_path = os.path.join(src_dir, 'best_config.json')

if os.path.exists(weights_path) and os.path.exists(config_path):
    model = NeuralNetwork.load(weights_path, config_path)
else:
    # Fallback to hardcoded architecture if best_config.json is missing
    best_config = argparse.Namespace(
        dataset='mnist',
        epochs=2,
        batch_size=64,
        loss='cross_entropy',
        optimizer='sgd',
        learning_rate=0.01,
        weight_decay=0.0,
        num_layers=2,
        hidden_size=[64, 64],
        activation='relu',
        weight_init='xavier'
    )
    model = NeuralNetwork(best_config)

    if os.path.exists(weights_path):
        weights = np.load(weights_path, allow_pickle=True).item()
        model.set_weights(weights)