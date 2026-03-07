from argparse import Namespace
from ann.neural_network import NeuralNetwork

try:
    args = Namespace(num_layers=2, hidden_size=[3, 2], activation="relu", loss="cross_entropy", weight_init="xavier")
    nn = NeuralNetwork(cli_args=args)
    print("Success with valid hidden_size")
except Exception as e:
    print("Error with valid hidden_size:", str(e))

try:
    args_none = Namespace(num_layers=2, hidden_size=None, activation="relu", loss="cross_entropy", weight_init="xavier")
    nn = NeuralNetwork(cli_args=args_none)
    print("Success with None hidden_size")
except Exception as e:
    print("Error with None hidden_size:", str(e))

