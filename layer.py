import math

import numpy as np

from node import Node


def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))


def softmax(vec):
	return np.exp(vec) / np.sum(np.exp(vec), axis=0)


def euclidean(activations, weights, dimension):
	if (dimension == 1):
		return [math.pow(np.linalg.norm(activations - w), 2) for w in weights]
	else:
		distances = []
		for i in range(len(weights)):
			distances.append([])
			for j in range(len(weights[i])):
				d = math.pow(np.linalg.norm(activations - weights[i][j]), 2)
				distances[i].append(d)
		return distances


class Layer():
	def __init__(self, function, input_size, output_size, dimension):
		self.function = function
		self.input_size = input_size
		self.output_size = output_size
		self.dimension = dimension
		self.init_nodes()

		# Initalizing nodes:
	def init_nodes(self):
		self.nodes = []
		self.weights = []
		if (self.dimension == 1):
			for i in range(self.output_size):
				node = Node(self.input_size)
				self.nodes.append(node)
				self.weights.append(node.weights)
		else:
			for i in range(self.output_size):
				self.nodes.append([])
				self.weights.append([])
				for j in range(self.output_size):
					node = Node(self.input_size)
					self.nodes[i].append(node)
					self.weights[i].append(node.weights)

	# Handling OUTPUT Activation of nodes in the given layer:
	# [Input_size x Ouput_size]-matrix ====> [Output_size]-Activation_Vector
	def compute_activation_vector(self, activations):
		if (self.function == 'euclidean'):
			return euclidean(activations, self.weights, self.dimension)


def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x, y)