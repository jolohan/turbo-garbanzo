import math

import numpy as np

import dynamic_plot
from data_manager import DataManager
from layer import Layer
import mnist_plot
import random


def manhattan(x, y):
	return abs(x[0] - y[0]) + abs(x[1] - y[1])


def euclidean(vec_1, vec_2):
	return math.sqrt(pow(vec_2[0] - vec_1[0], 2) + pow(vec_2[1] - vec_1[1], 2))


class Network():
	def __init__(self, input_activation_func='euclidean', output_activation_func='euclidean',
	             input_size=2, output_size=100, epochs=40, learning_rate=0.01, learning_decay=1.0,
	             initial_neighborhood=20, neighborhood_decay=0.5, data_manager=None, node_multiplier=1,
	             dimension=1):
		if data_manager == None:
			self.data_manager = DataManager(1)
		else:
			self.data_manager = data_manager
		self.dimension = dimension
		self.input_size = self.data_manager.input_size
		self.output_size = self.data_manager.output_size*node_multiplier
		print(self.input_size, self.output_size)

		# Data:
		self.input = self.data_manager.input
		if (dimension == 2):
			self.labels = self.data_manager.labels

		# Parameters:
		self.epochs = epochs
		self.similarity = 0.5
		self.initial_learning_rate = learning_rate
		if (self.dimension == 1):
			self.initial_neighborhood = int(self.output_size*0.1)
		else:
			self.initial_neighborhood = int(self.output_size*self.output_size*0.05)
		print("Initial Neighborhood size = " + str(self.initial_neighborhood))
		self.learning_decay = learning_decay
		self.neighborhood_decay = neighborhood_decay

		# Need only 1 layer!
		self.node_layer = Layer(output_activation_func, self.input_size, self.output_size, dimension=dimension)

	def forward(self, input):
		output = self.node_layer.compute_activation_vector(input)
		return output

	def run_once(self, input):
		activations = self.forward(input)
		if (self.dimension == 1):
			return np.argmin(activations), activations[np.argmin(activations)]
		else:
			index = np.argmin(activations)
			first = index//len(activations)
			second = index % len(activations)
			
			#print("index = " + str(index) + ", First = " + str(first) + ", second = " + str(second))
			return (first, second), activations[first][second]

	def classify(self):
		print("Classifying nodes...")
		for image, label in zip(self.input, self.labels):
			winning_index, _ = self.run_once(image)
			self.node_layer.nodes[winning_index[0]][winning_index[1]].class_labels[label + 1] += 1

		for n_list in self.node_layer.nodes:
			for n in n_list:
				n.label = np.argmax(n.class_labels)

	def train(self):
		print("Starting training on " + str(len(self.input)) + " samples.")
		PLOT_EVERY = 10
		PRINT_EVERY = 10
		CLASSIFY_EVERY = 40
		FIRST = 100
		avg_loss = 0.0
		old_distance = 9999999999.9
		converge_flag = 0
		test_sample = self.input[0]
		for t in range(self.epochs):
			train_index = np.random.choice(len(self.input), 1)[0]
			train_sample = self.input[train_index]
			#print("Image: ", train_sample)
			winning_index, loss = self.run_once(train_sample)

			self.optimize_network(winning_index, train_sample, t)
			avg_loss += loss
			if (t >= FIRST and t % CLASSIFY_EVERY == 0 and self.dimension == 2):
				self.test("test", False)
			if (t % PLOT_EVERY == 0):
				if (self.dimension == 1):
					dynamic_plot.plot_map(self.input, self.get_weights(), t, self.data_manager.file)
				else:
					#mnist_plot.plot(self.forward(test_sample), t)
					pass
					#mnist_plot.plot_labels(self.node_layer.nodes, t)
			if (t > 0 and t % PRINT_EVERY == 0):
				print("\nTraining Epoch " + str(t) + "/" + str(self.epochs))
				print("Avg Loss = " + str(avg_loss / t))
				if (self.dimension == 1):
					current_distance = self.calculate_tsp_distance()
					print("TSP Distance = " + str(current_distance))
					dynamic_plot.plot_map(self.input, self.get_weights(), t, self.data_manager.file)
					if (current_distance == old_distance):
						converge_flag += 1
					else:
						converge_flag = 0
					old_distance = current_distance
					if converge_flag > 10:
						break

	# Required method for testing on both the training set and test set:
	def test(self, dataset, plot):
		self.classify()
		tr = False
		if (dataset == "train"):
			print("\n--- Training Set Test ---\n")
			data = self.input
			labels = self.labels
			tr = True
		else:
			print("\n--- Test Set Test ---\n")
			data = self.data_manager.test_input
			labels = self.data_manager.test_labels

		windexes = []
		correct = 0
		total = 0
		not_class = 0
		for image, label in zip(data, labels):
			winning_index, _ = self.run_once(image)
			node_guess = int(self.node_layer.nodes[winning_index[0]][winning_index[1]].label)
			if node_guess == 0:
				not_class += 1
				continue
			total += 1
			if (node_guess - 1 == label):
				correct += 1
			windexes.append([winning_index, label])
		print("\nPercentage correct classifications = " + str(float(100.0*correct/max(1, total))) + " %")
		print("\nPercentage not classified = " + str(float(100.0*not_class/max(1, not_class +  total))) + " %")
		if (plot):
			mnist_plot.plot_winners(windexes, tr)


	def get_weights_to(self, i, j=None):
		if (self.dimension == 1):
			return self.node_layer.weights[i]
		else:
			return self.node_layer.weights[i][j]

	def get_weights(self):
		return self.node_layer.weights

	def update_neighbors(self, winning_index, t, learning_rate, input_values):
		# Decay neighborhood size:
		neighborhood_size = self.initial_neighborhood * math.exp(((-1.0 * t) / self.neighborhood_decay))

		# Needs weights as either a 2D list, or a 1D list (of weights):
		weights = self.get_weights()

		# 1D SOM:
		if (self.dimension == 1):
			# Compute the T matrix for neighbor updates:
			for j in range(len(weights)):
				d = abs(winning_index - j)
				distance = min(d, len(weights) - d)
				self.node_layer.weights[j] += learning_rate * self.compute_neighborhood(distance, neighborhood_size) * (input_values - self.node_layer.weights[j])
		# 2D SOM:
		else:
			ds = []
			for i in range(len(weights)):
				ds.append([])
				for j in range(len(weights[i])):
					distance = euclidean([i, j], winning_index)
					ds[i].append(int(distance))
					self.node_layer.weights[i][j] += learning_rate * self.compute_neighborhood(distance, neighborhood_size) * (input_values - self.node_layer.weights[i][j])

	def compute_neighborhood(self, distance, size):
		area = size ** 2
		if (area == 0):
			return 0
		return math.exp(-distance ** 2 / (2 * area))

	def optimize_network(self, j, input_values, t):
		# Decay Learning rate:
		learning_rate = self.initial_learning_rate * math.exp(((-1.0 * t) / self.learning_decay))
		self.update_neighbors(j, t, learning_rate, input_values)


	def calculate_tsp_distance(self):
		nodes = self.node_layer.nodes
		city_nodes = {}
		for city in self.input:
			# Find the best node:
			# Maybe try neighbouring nodes if best node already has a city
			index, _ = self.run_once(city)
			if index not in city_nodes:
				city_nodes[index] = [city]
			else:
				city_nodes[index].append(city)
		# Reorder nodes after city-nodes:
		tsp_order = []
		for node_index in range(len(nodes)):
			if node_index in city_nodes:
				#tsp_order.append(self.get_weights_to(node_index))
				random_index = random.randrange(len(city_nodes[node_index]))
				if (len(city_nodes[node_index]) > 1):
					print("some nodes point to several citites")
				tsp_order.append(city_nodes[node_index][random_index])

		# Calculate the total distance:
		tsp_distance = euclidean(tsp_order[0], tsp_order[-1])
		for index in range(len(tsp_order) - 1):
			tsp_distance += euclidean(tsp_order[index], tsp_order[index + 1])

		return tsp_distance * self.data_manager.norm_constant
