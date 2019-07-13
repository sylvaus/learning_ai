from typing import List, Callable

import numpy as np


def sigmoid(val: float):
    return 1.0 / (1.0 + np.exp(-val))


def relu(val: float):
    return np.maximum(0.0, val)


class FastNeuralNetwork:
    def __init__(
            self
            , layers_weights: List[np.ndarray] = None
            , layers_biases: List[np.ndarray] = None
            , layers_functions: List[Callable] = None
    ):
        if layers_weights is None:
            layers_weights = []
        self._layers_weights = layers_weights

        if layers_biases is None:
            layers_biases = [np.zeros((layers_weights.shape[0], 0)) for layers_weights in layers_weights]
        self._layers_biases = layers_biases

        if layers_functions is None:
            layers_functions = [relu for _ in layers_weights]
            if layers_functions:
                layers_functions[-1] = sigmoid
        self._layers_functions = layers_functions

    def add_layer(self, weights: np.ndarray, biases: np.ndarray = None, func: Callable = relu):
        if self._layers_weights:
            assert self._layers_weights[-1].shape[0] == weights.shape[1]\
                , "Weights are not compatible with previous layer: expected {} columns, got {}"\
                  .format(self._layers_weights[-1].shape[0], weights.shape[1])
        if biases:
            assert biases.shape[0] == weights.shape[0]\
                , "Weights are not compatible with biases: expected {} columns, got {}"\
                  .format(weights.shape[0], biases.shape[0])
        else:
            biases = np.zeros((weights.shape[0], 1))
        self._layers_weights.append(weights)
        self._layers_biases.append(biases)
        self._layers_functions.append(func)

    def feedfordward(self, input_array: np.ndarray):
        output = input_array
        for weights, biases, func in zip(self._layers_weights, self._layers_biases, self._layers_functions):
            output = func(weights.dot(output) + biases)

        return output

    def copy(self, deepcopy=False):

        if deepcopy:
            return FastNeuralNetwork(
                self._layers_weights.copy()
                , self._layers_biases.copy()
                , self._layers_functions.copy()
            )
        else:
            return FastNeuralNetwork(
                self._layers_weights.copy()
                , self._layers_biases.copy()
                , self._layers_functions.copy()
            )