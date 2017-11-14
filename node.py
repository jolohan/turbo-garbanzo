import numpy as np
import matplotlib.pyplot as plt

class Node():
	
	def __init__(self, output_size, weight_interval=[-1.0, 1.0]):
		self.init_weights(output_size, weight_interval)
		self.activation = 0.0


	# Handling Weights:
	def init_weights(self, output_size, weight_interval):
		self.weights = np.random.uniform(low=weight_interval[0], high=weight_interval[1], size=output_size)

	def plot_weights(self):
		count, bins, ignored = plt.hist(self.weights, 15, normed=True)
		plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
		plt.show()

	# Handling activation:
	def set_activation(self, activation):
		self.activation = activation