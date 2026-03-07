import numpy as np 
import argparse
from ann.neural_network import NeuralNetwork

best_config  = argparse.Namespace(
    dataset = 'mnist',
    epochs = 2,
    batch_size = 64,
    loss = 'cross_entropy',
    optimizer = 'sgd',
    learning_rate = 0.01,
    weight_decay = 0.0,
    num_layers = 2,
    hidden_size = [64,64],
    activation = 'relu',
    weight_init = 'xavier'
)

model = NeuralNetwork(best_config)

weights = np.load('best_model.npy',allow_pickle=True).item()
model.set_weights(weights)