import math
import numpy as np
import random
import dynamic_plot
from data_manager import DataManager
from layer import Layer


def manhattan(x, y):
	return abs(x[0] - y[0]) + abs(x[1] - y[1])


def euclidean(vec_1, vec_2):
	return math.sqrt(pow(vec_2[0]-vec_1[0],2) + pow(vec_2[1]-vec_1[1],2))


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
    	activations = self.forward(input)
        return np.argmin(activations), activations[np.argmin(activations)]

    def train(self):
        test_sample = self.input[0]
        PRINT_EVERY = 10
        avg_loss = 0.0
        for t in range(self.epochs):
            train_index = np.random.choice(len(self.input), 1)[0]
            train_sample = self.input[train_index]
            winning_index, loss = self.run_once(train_sample)
            self.optimize_network(winning_index, train_sample, t)
            avg_loss += loss
            if (t > 0 and t % PRINT_EVERY == 0):
           	    print("\nTraining Epoch " + str(t))
           	    print("Avg Loss = " + str(avg_loss/t))
           	    current_distance = self.calculate_tsp_distance()
           	    print("TSP Distance = " + str(current_distance))
           	    dynamic_plot.plot_map(self.input, self.get_weights(), t, self.data_manager.file)

    def get_weights_to(self, j):
        return self.input_layer.get_out_weights(j)

    def get_weights(self):
        return [self.input_layer.get_out_weights(j) for j in range(self.output_size)]

    def get_best_neighbors(self, index, t):
    	# Decay neighborhood size:
        neighborhood_size = self.initial_neighborhood * math.exp(((-1.0*t) / self.neighborhood_decay))
        weights = self.get_weights()
        T_matrix = []

        # Compute the T matrix for neighbor updates:
        for j in range(len(weights)):
            d = abs(index - j)
            distance = min(d, len(weights)-d)
            T_matrix.append(self.compute_neighborhood(distance, neighborhood_size))
        sorted_matrix = sorted(((value, index) for index, value in enumerate(T_matrix)), reverse=False)
        return sorted_matrix

    def compute_neighborhood(self, distance, size):
        area = size**2
        if (area == 0):
            return 0
        return math.exp(-distance**2/(2*area))


    def optimize_network(self, j, input_values, t):
    	# Decay Learning rate:
        learning_rate = self.initial_learning_rate * math.exp(((-1.0*t)/ self.learning_decay))
        neighbors = self.get_best_neighbors(j, t)

        # Update all Neighbors:
        for t_val, k in neighbors:
            for i, n in enumerate(self.input_layer.nodes):
                update = learning_rate * t_val * (input_values[i] - n.weights[k])
                n.weights[k] += update

    def calculate_tsp_distance(self):
        nodes = self.output_layer.nodes
        city_nodes = {}
        for city in self.input:
            # Find the best node:
            index, _ = self.run_once(city)
            if index not in city_nodes:
                city_nodes[index] = [city]
            else:
                city_nodes[index].append(city)

        # Reorder nodes after city-nodes:
        tsp_order = []
        for node_index in range(len(nodes)):
            if node_index in city_nodes:
                tsp_order.append(self.get_weights_to(node_index))

        # Calculate the total distance:
        tsp_distance = euclidean(tsp_order[0], tsp_order[-1])
        for index in range(len(tsp_order)-1):
            tsp_distance += euclidean(tsp_order[index], tsp_order[index + 1])

        return tsp_distance*self.data_manager.norm_constant
