from typing import List, Callable

import numpy as np


def sigmoid(val: float):
    return 1.0 / (1.0 + np.exp(-val))

def deriv_sigmoid(val: float):
    return sigmoid(val) * (1 - sigmoid(val))

def relu(val: float):
    return np.maximum(0.0, val)

def cost_derivative(activation, expected):
    return activation - expected


class NeuralNetwork:
    def __init__(
            self
            , layers_weights: List[np.ndarray] = None
            , layers_biases: List[np.ndarray] = None
            , activation_func: Callable = None
            , deriv_activation_func: Callable = None
            , cost_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    ):
        if layers_weights is None:
            layers_weights = []
        self._layers_weights = layers_weights

        if layers_biases is None:
            layers_biases = [np.zeros((layers_weights.shape[0], 0)) for layers_weights in layers_weights]
        self._layers_biases = layers_biases

        if activation_func is None:
            activation_func = sigmoid
        self._activation_func = activation_func

        if deriv_activation_func is None:
            deriv_activation_func = deriv_sigmoid
        self._deriv_activation_func = deriv_activation_func

    def add_layer(self, weights: np.ndarray, biases: np.ndarray = None):
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

    def add_rand_layer(self, input_nb, output_nb, min_weight=-1.0, max_weight=-1.0, min_bias=-1.0, max_bias=-1.0):
        self.add_layer(
            np.random.uniform(low=min_weight, high=max_weight, size=(output_nb, input_nb))
            , np.random.uniform(low=min_bias, high=max_bias, size=(output_nb, 1))
        )

    def feedfordward(self, input_array: np.ndarray):
        output = input_array
        for weights, biases in zip(self._layers_weights, self._layers_biases):
            output = self._activation_func(weights.dot(output) + biases)

        return output

    def backpropagation(self, input_array: np.ndarray, output_array: np.ndarray, rate: float):
        neuron_values = []
        activation = input_array
        activations = [input_array]

        for weights, biases in zip(self._layers_weights, self._layers_biases):
            neuron_value = weights.dot(activation) + biases
            neuron_values.append(neuron_value)
            activation = self._activation_func(neuron_value)
            activations.append(activation)

        error = np.multiply(cost_derivative(activation, output_array), self._deriv_activation_func(neuron_values[-1]))
        self._layers_weights[-1] = self._layers_weights[-1] - rate * np.dot(error, activation[-2].transpose())
        self._layers_biases[-1] = self._layers_biases[-1] - rate * error
        for i in range(2, len(self._layers_weights)):
            error = np.multiply(
                self._layers_weights[-i+1].transpose() * error
                , self._deriv_activation_func(neuron_values[-i])
            )
            self._layers_weights[-i] = self._layers_weights[-1] - rate * np.dot(error, activation[-i-1].transpose())
            self._layers_biases[-i] = self._layers_biases[-1] - rate * error


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_rand_layer(2, 2)
    nn.add_rand_layer(2, 1)

    nn.backpropagation(np.array([1, 0]).transpose(), np.array([1]), 0.1)


