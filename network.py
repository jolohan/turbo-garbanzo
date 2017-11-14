from layer import Layer
from data_manager import DataManager
import math
import numpy as np
import random


def manhattan(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

class Network():
    def __init__(self, input_activation_func='euclidean', output_activation_func='euclidean',
                 input_size=2, output_size=100, epochs=40, learning_rate=0.01, learning_decay=1.0,
                 initial_neighborhood=20, neighborhood_decay=0.5, data_manager=None):
        if data_manager == None:
            self.data_manager = DataManager(1)
        else:
            self.data_manager = data_manager
        self.input_size = self.data_manager.input_size
        self.output_size = self.data_manager.output_size
        print(self.input_size, self.output_size)
        self.input = self.data_manager.input
        self.epochs = epochs
        self.similarity = 0.5
        self.initial_learning_rate = learning_rate
        self.initial_neighborhood = initial_neighborhood
        self.learning_decay = learning_decay
        self.neighborhood_decay = neighborhood_decay
        self.input_layer = Layer(input_activation_func, self.input_size, self.output_size)
        self.output_layer = Layer(output_activation_func, self.output_size, output_size=0, output_layer=True)

    def forward(self, input):
        output = self.input_layer.compute_activation_vector(input)
        return output

    def run_once(self, input):
        return np.argmin(self.forward(input))

    def train(self):
        test_sample = self.input[0]
        print("Coordinated: ", test_sample)
        index = self.run_once(test_sample)
        print("Inital weights: ", self.get_weights_to(index))
        for t in range(self.epochs):
            print("\nTraining Epoch " + str(t))
            for i in range(len(self.input)):
                train_index = np.random.choice(len(self.input), 1)
                train_sample = self.input[train_index[0]]
                index = self.run_once(train_sample)
                self.optimize_network(index, train_sample, t)
        print("Coordinated: ", test_sample)
        index = self.run_once(test_sample)
        print("Final weights: ", self.get_weights_to(index))

    def get_weights_to(self, j):
        return self.input_layer.get_out_weights(j)

    def get_weights(self):
        return [self.input_layer.get_out_weights(j) for j in range(self.output_size)]

    def get_best_neighbors(self, index, t):
        neighborhood_size = int(self.initial_neighborhood * math.exp(-t / self.neighborhood_decay))
        weights = self.get_weights()
        T_matrix = [math.exp(
            -(math.pow(manhattan(weights[j], weights[index]), 2)) / max(1, (2 * math.pow(neighborhood_size, 2))))
                    for j in range(len(weights))]
        sorted_matrix = sorted(((value, index) for index, value in enumerate(T_matrix)), reverse=True)
        return sorted_matrix[:neighborhood_size]

    def optimize_network(self, j, input_values, t):
        learning_rate = self.initial_learning_rate * math.exp(-t / self.learning_decay)
        for i, n in enumerate(self.input_layer.nodes):
            n.weights[j] += learning_rate * (input_values[i] - n.weights[j])
        neighbors = self.get_best_neighbors(j, t)
        for val, k in neighbors:
            for i, n in enumerate(self.input_layer.nodes):
                n.weights[k] += learning_rate * self.similarity * (input_values[i] - n.weights[k])
