import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataManager
from mnist_manager import MNIST
from network import Network
import os


class Display():
	def __init__(self, network):
		self.network = network
		plt.ion()

	def plot_output_weights(self, fig=None, title='TSP_solution'):
		weights = self.network.get_weights()
		weights = np.array(weights)
		cities = self.network.data_manager.input
		fig = fig if fig else plt.figure()
		axes = fig.gca()
		axes.clear()
		x = weights[:, 0]
		y = weights[:, 1]
		# print(x)
		plt.scatter(x, y, c="blue", alpha=0.5, marker='x', label="Chosen path")
		plt.pause(0.001)
		for i in range(len(weights) - 1):
			x1 = x[i]
			x2 = x[i + 1]
			# print(x2)
			y1 = y[i]
			y2 = y[i + 1]
			plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=0.2)
		plt.plot([x[0], x[-1]], [y[0], y[-1]], color='k', linestyle='-', linewidth=0.2)
		plt.pause(0.001)
		a = cities[:, 0]
		b = cities[:, 1]
		plt.scatter(a, b, c="red", alpha=0.5, marker='o', label="City")
		plt.pause(0.001)

		plt.legend(loc=2)
		axes.autoscale_view()
		plt.draw()
		fig.savefig("plots/tsp/" + title + ".png")
		plt.pause(0.01)
		return fig

		
class Interface():

	def __init__(self, config_file='config/TSP_config.txt'):

		print("\n--- SOM Module Interface ---\n")

		# Creating dictionary from all config files:
		path = "config/"
		config_dictionary = {}
		index = 1
		for subdirs, dirs, files in os.walk(path):
			for file in files:
				if ('.txt' in file):
					config_dictionary[index] = file
					index += 1
		config_dictionary[0] = 'Exit'

		finished = False

		while not finished:
			for key in sorted(config_dictionary.keys()):
				print(str(key) + ": ", config_dictionary[key])

			config = input("\nWhich config to run [0/" + str(len(config_dictionary)-1) + "]: ")
			config_nr = int(config)

			if (config_nr == 0):
				finished = True
				break
		
			self.load_config(filename=path + config_dictionary[config_nr])
			if (self.problem_type == "TSP"):
				self.data_manager = DataManager(self.problem_number)
				self.dimension = 1
			else:
				self.data_manager = MNIST(self.size, self.test_size, self.out_size)
				self.dimension = 2

			network = Network(epochs=self.epochs, learning_rate=self.learning_rate,
							  learning_decay=self.learning_decay, initial_neighborhood=self.initial_neighbourhood,
							  neighborhood_decay=self.neighbourhood_decay, data_manager=self.data_manager,
			                  learning_rate_constant=self.learning_rate_constant, node_multiplier=self.node_multiplier,
			                  dimension = self.dimension)
			
			network.start_epoch = 0
			while (network.epochs > network.start_epoch):
				network.train()
				if (self.dimension == 1):
					pass
					#fig = display.plot_output_weights(fig)
				else:
					network.test("train", True)
					network.test("test", True)
				input_text = "abc"
				while (input_text != "STOP"):
					input_text = input("How many more epochs do you want to train? 0 to quit. ")
					try:
						network.start_epoch = network.epochs
						network.epochs = network.epochs + int(input_text)
						break
					except:
						if (input_text.lower() == "q" or input_text == "0"):
							network.epochs = 0
							break
						print("Fail. Enter a number")

			done = input("Exit = 0, continue = 1: ")

			try:
				done = int(done)
				if (not done):
					finished = True
					break

			except:
				print("Continuing...")
		print("Exiting...")

	def load_config(self, filename):

		# Pre-processing config file:
		network_dict = {}
		with open(filename, "r") as file:
			for line in file:
				listed_specs = line.split(" ")
				network_dict[listed_specs[0]] = [item.strip() for item in listed_specs[1:]]

		# Parameters for output generation:
		self.problem_type = network_dict['Problem'][0]

		if (self.problem_type == 'TSP'):
			self.problem_number = network_dict['Problem'][1]
		else:
			self.size = float(network_dict['Problem'][1])
			self.test_size = float(network_dict['Problem'][2])
			self.out_size = int(network_dict['Size'][0])

		# 1. Epochs (Total number of MINIBATCHES during training)
		self.epochs = int(network_dict['Epochs'][0])

		# 2. Learning Rate
		self.learning_rate = float(network_dict['LearningRate'][0])

		# 3. Learning Decay
		self.learning_decay = float(network_dict['LearningDecay'][0])

		# 4. Initial Neighbourhood
		self.initial_neighbourhood = float(network_dict['InitialNeighbourhood'][0])

		# 5. Neighbourhood decay
		self.neighbourhood_decay = float(network_dict['NeighbourhoodDecay'][0])

		# 6. Learning Rate Constant
		self.learning_rate_constant = float(network_dict['LearningRateConstant'][0])

		# 7. Node Multiplier
		self.node_multiplier = int(network_dict['NodeMultiplier'][0])
		


if __name__ == '__main__':
	Interface()
