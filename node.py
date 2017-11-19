import matplotlib.pyplot as plt
import numpy as np


class Node():
	def __init__(self, input_size, weight_interval=[0.0, 1.0], weights=None):
		self.init_weights(input_size, weight_interval, weights)
		self.activation = 0.0

	# Handling Weights:
	def init_weights(self, input_size, weight_interval, weights=None):
		if (weights==None):
			self.weights = np.random.uniform(low=weight_interval[0], high=weight_interval[1], size=input_size)
		else:
			self.weights = weights

	def plot_weights(self):
		count, bins, ignored = plt.hist(self.weights, 15, normed=True)
		plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
		plt.show()

	# Handling activation:
	def set_activation(self, activation):
		self.activation = activation
