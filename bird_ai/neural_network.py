from copy import deepcopy
from random import uniform
from time import time

from math import exp
from typing import Union, Tuple, Iterable, List, Callable

import numpy as np
from numpy.random.mtrand import randint


def sigmoid(val: float):
    return 1.0 / (1.0 + np.exp(-val))


def relu(val: float):
    return np.maximum(0.0, val)


class InputNeuron:
    def __init__(self, initial_value=0):
        self.value = initial_value


class SigmoidNeuron:
    def __init__(self, bias=0):
        self._bias = bias
        self._weight_inputs = []

    @property
    def weights(self):
        return (
            weight_input[0]
            for weight_input in self._weight_inputs
        )

    @property
    def value(self):
        value = 0
        for weight, input_ in self._weight_inputs:
            value += weight * input_.value

        return sigmoid(value)

    def connect_inputs(self, weight_inputs: Iterable[Tuple[float, Union["SigmoidNeuron", InputNeuron]]]):
        self._weight_inputs.extend(weight_inputs)


class ReluNeuron:
    def __init__(self, bias=0):
        self._bias = bias
        self._weight_inputs = []

    @property
    def weights(self):
        return (
            weight_input[0]
            for weight_input in self._weight_inputs
        )

    @property
    def value(self):
        value = 0
        for weight, input_ in self._weight_inputs:
            value += weight * input_.value

        return max(0, value)

    def connect_inputs(self, weight_inputs: Iterable[Tuple[float, Union["SigmoidNeuron", InputNeuron]]]):
        self._weight_inputs.extend(weight_inputs)


class NeuralNetwork:
    def __init__(self, nb_inputs):
        self._inputs = [InputNeuron() for _ in range(nb_inputs)]
        self._hidden_layers = []
        self._output = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def hidden_layers_weights(self):
        return [
            [
                neuron.weights
                for neuron in layer
            ]
            for layer in self._hidden_layers
        ]

    @property
    def output(self):
        return self._output

    @property
    def output_weights(self):
        return self._output.weights

    def add_layer(self, neuron_weights):
        if self._hidden_layers:
            previous_layer = self._hidden_layers[-1]
        else:
            previous_layer = self._inputs
        layer = []
        for neuron_weight in neuron_weights:
            assert len(neuron_weight) == len(previous_layer) \
                , "Not enough weights for the number of neurons in the previous layer"

            neuron = ReluNeuron()
            neuron.connect_inputs(zip(neuron_weight, previous_layer))

            layer.append(neuron)

        self._hidden_layers.append(layer)

    def add_output(self, neuron_weights):
        previous_layer = self._hidden_layers[-1]
        assert len(neuron_weights) == len(previous_layer) \
            , "Not enough weights for the number of neurons in the previous layer"

        neuron = SigmoidNeuron()
        neuron.connect_inputs(zip(neuron_weights, previous_layer))

        self._output = neuron


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

    def mutate(
            self, nb_weight_mutations, nb_bias_mutations
            , weight_mutation=lambda: uniform(-0.1, 0.1)
            , bias_mutation=lambda: uniform(-0.1, 0.1)):

        layers_weights = deepcopy(self._layers_weights)
        for _ in range(nb_weight_mutations):
            layer = randint(0, len(layers_weights) - 1)
            row = randint(0, layers_weights[layer].shape[0] - 1)
            col = randint(0, layers_weights[layer].shape[1] - 1)
            layers_weights[layer][row][col] += weight_mutation()

        layers_biases = deepcopy(self._layers_biases)
        for _ in range(nb_bias_mutations):
            layer = randint(0, len(layers_biases) - 1)
            row = randint(0, layers_biases[layer].shape[0] - 1)
            layers_biases[layer][row][0] += bias_mutation()

        return FastNeuralNetwork(layers_weights, layers_biases, self._layers_functions)



def create_random_layer(previous_layer_size, layer_size):
    return list([
        tuple(uniform(-1, 1) for _ in range(previous_layer_size))
        for _ in range(layer_size)
    ])


if __name__ == '__main__':
    in1 = InputNeuron(1)
    in2 = InputNeuron(1)

    s1 = SigmoidNeuron()
    s1.connect_inputs([(0.8, in1), (0.2, in2)])
    s2 = SigmoidNeuron()
    s2.connect_inputs([(0.4, in1), (0.9, in2)])
    s3 = SigmoidNeuron()
    s3.connect_inputs([(0.3, in1), (0.5, in2)])

    e = SigmoidNeuron()
    e.connect_inputs([(0.3, s1), (0.5, s2), (0.9, s3)])

    print(e.value)
    values = [(uniform(-10, 10), uniform(-10, 10)) for _ in range(80000)]

    first_layer = create_random_layer(2, 10)
    second_layer = create_random_layer(10, 10)
    last_layer = create_random_layer(10, 1)
    start = time()
    result = 0
    nn = FastNeuralNetwork()
    nn.add_layer(np.array(first_layer))
    nn.add_layer(np.array(second_layer))
    nn.add_layer(np.array(last_layer), func=sigmoid)
    for value1, value2 in values:
        result += nn.feedfordward(np.array([[value1], [value2]]))[0][0]

    print(time() - start, result)
    nn.mutate(10, 10)

    start = time()
    result = 0
    nn = NeuralNetwork(2)
    nn.add_layer(first_layer)
    nn.add_layer(second_layer)
    nn.add_output(last_layer[0])
    for value1, value2 in values:
        nn.inputs[0].value = value1
        nn.inputs[1].value = value2
        result += nn.output.value

    print(time() - start, result)
