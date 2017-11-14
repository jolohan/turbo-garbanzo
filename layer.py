import numpy as np
from node import Node
import math

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

def softmax(vec):
	return np.exp(vec)/np.sum(np.exp(vec), axis=0)


class Layer():

	def __init__(self, function, input_size, output_size, output_layer=False):
		self.function = function
		self.input_size = input_size
		self.output_size = output_size
		self.output_layer = output_layer
		self.init_nodes()

	# Initalizing nodes:
	def init_nodes(self):
		self.nodes = [Node(self.output_size, self.output_layer) for i in range(self.input_size)]

	# Handling OUTPUT Activation of nodes in the given layer:
	# [Input_size x Ouput_size]-matrix ====> [Output_size]-Activation_Vector
	def compute_activation_vector(self, activations):
		weight_vector = [n.weights for n in self.nodes]
		activation_vector = np.dot(activations, weight_vector)
		if (self.function == 'linear'):
			return activation_vector
		elif (self.function == 'relu'):
			return [max(a, 0) for a in activation_vector]
		elif (self.function == 'sigmoid'):
			return [sigmoid(a) for a in activation_vector]
		elif (self.function == 'softmax'):
			return softmax(activation_vector)
		else:
			return activation_vector

