import math

import numpy as np

from node import Node


def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))


def softmax(vec):
	return np.exp(vec) / np.sum(np.exp(vec), axis=0)


def euclidean(activations, weights):
	return [math.pow(np.linalg.norm(activations - weights[:, j]), 2) for j in range(weights.shape[1])]


class Layer():
	def __init__(self, function, input_size, output_size, output_layer=False):
		self.function = function
		self.input_size = input_size
		self.output_size = output_size
		self.output_layer = output_layer
		self.init_nodes()

		# Initalizing nodes:
	def init_nodes(self):
		"""
		radius = 0.1
		self.nodes = []
		x = []
		y = []
		for i in range(self.input_size):
			phi = i * 3.14 / self.input_size
			#print("Phi:", phi)
			weights = pol2cart(radius, phi)
			x.append(weights[0])
			y.append(weights[1])
		node = Node(self.output_size, self.output_layer, weights=x)
		self.nodes.append(node)
		node = Node(self.output_size, self.output_layer, weights=y)
		self.nodes.append(node)
		print("Weights:", node.weights)
		self.nodes.append(node)
		step = 1.0/(self.input_size**2)
		self.nodes = [Node(self.output_size, self.output_layer,
		                   weight_interval=[step*i, step*(i+1)]) for i in range(self.input_size)]
		"""
		self.nodes = [Node(self.output_size, self.output_layer) for i in range(self.input_size)]


	# Handling OUTPUT Activation of nodes in the given layer:
	# [Input_size x Ouput_size]-matrix ====> [Output_size]-Activation_Vector
	def compute_activation_vector(self, activations):
		weight_vector = np.array([n.weights for n in self.nodes])
		if (self.function == 'euclidean'):
			return euclidean(activations, weight_vector)
		"""
		elif (self.function == 'relu'):
			return [max(a, 0) for a in activation_vector]
		elif (self.function == 'sigmoid'):
			return [sigmoid(a) for a in activation_vector]
		elif (self.function == 'softmax'):
			return softmax(activation_vector)
		else:
			return activation_vector
		"""

	def get_out_weights(self, j):
		return [n.weights[j] for n in self.nodes]

def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x, y)