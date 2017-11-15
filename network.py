import math

import numpy as np

from data_manager import DataManager
from layer import Layer


def manhattan(x, y):
	return abs(x[0] - y[0]) + abs(x[1] - y[1])


def euclidean(vec_1, vec_2):
	return math.pow(np.linalg.norm(vec_1 - vec_2), 2)


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
			train_indexes = np.random.choice(len(self.input), len(self.input), replace=False)
			for i in train_indexes:
				train_sample = self.input[train_indexes[i]]
				train_index, loss = self.run_once(train_sample)
				self.optimize_network(train_index, train_sample, t)
				avg_loss += loss
			if (t > 0 and t % PRINT_EVERY == 0):
				print("Training Epoch " + str(t))
				print("Avg Loss = " + str(avg_loss / t))
				current_distance = self.calculate_tsp_distance()
				print("TSP Distance = " + str(current_distance))

	def get_weights_to(self, j):
		return self.input_layer.get_out_weights(j)

	def get_weights(self):
		return [self.input_layer.get_out_weights(j) for j in range(self.output_size)]

	def get_best_neighbors(self, index, t):
		neighborhood_size = self.initial_neighborhood * math.exp(((-1.0 * t) / self.neighborhood_decay))
		# if (t % 10 == 0):
		#	print("neighborhood_size = " + str(neighborhood_size))
		weights = self.get_weights()
		T_matrix = []
		for j in range(len(weights)):
			d = index - j
			if (d > (len(weights) / 2.0)):
				distance = math.pow(abs((index - len(weights)) - j), 2)
			else:
				distance = math.pow(abs(index - (len(weights) - j)), 2)
			# distance = math.pow(abs(index - j), 2)
			# distance = math.pow(manhattan(weights[]))
			divisor = (2 * math.pow(neighborhood_size, 2))
			T_matrix.append(math.exp((-1.0 * (distance / max(1.0, divisor)))))
		sorted_matrix = sorted(((value, index) for index, value in enumerate(T_matrix)), reverse=False)
		return sorted_matrix  # [:max(1, int(neighborhood_size))]

	def optimize_network(self, j, input_values, t):
		learning_rate = self.initial_learning_rate * math.exp(((-1.0 * t) / self.learning_decay))
		# if (t % 10 == 0):
		#	print("Learning rate = " + str(learning_rate))
		for i, n in enumerate(self.input_layer.nodes):
			# print(n.weights[j])
			n.weights[j] += learning_rate * (input_values[i] - n.weights[j])
			# print("After: ", n.weights[j])
		neighbors = self.get_best_neighbors(j, t)
		for t_val, k in neighbors:
			if (k == j):
				continue
			for i, n in enumerate(self.input_layer.nodes):
				update = learning_rate * t_val * (input_values[i] - n.weights[k])
				n.weights[k] += update

	def calculate_tsp_distance(self):
		cities = self.input
		nodes = self.output_layer.nodes
		city_nodes = {}
		for city_idx, city in enumerate(cities):
			# Find the best node:
			idx, _ = self.run_once(city)
			if idx not in city_nodes:
				city_nodes[idx] = [city]
			else:
				city_nodes[idx].append(city)

		# Reorder nodes after city-nodes:
		tsp_order = []
		for node_index in range(len(nodes)):
			if node_index in city_nodes:
				tsp_order += self.get_weights_to(node_index)

		# Calculate the total distance:
		tsp_distance = euclidean(tsp_order[0], tsp_order[-1])
		for idx in range(len(tsp_order) - 1):
			tsp_distance += euclidean(tsp_order[idx], tsp_order[idx + 1])

		return tsp_distance
