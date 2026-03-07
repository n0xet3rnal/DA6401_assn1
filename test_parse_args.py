from argparse import Namespace
from ann.neural_network import NeuralNetwork

def test_missing_args():
    try:
        # What if it's missing num_layers?
        args = Namespace(hidden_size=[3, 2], activation="relu", loss="cross_entropy", weight_init="xavier")
        nn = NeuralNetwork(cli_args=args)
        print("Success missing num_layers")
    except Exception as e:
        print("Error missing num_layers:", str(e))

test_missing_args()
