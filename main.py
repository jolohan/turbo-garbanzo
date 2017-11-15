import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataManager
from network import Network


class Display():
	def __init__(self, network):
		self.network = network
		plt.ion()

	def plot_output_weights(self, fig=None, title='tits'):
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

		plt.xlabel("Leprechauns")
		plt.ylabel("Gold")
		plt.legend(loc=2)
		axes.autoscale_view()
		plt.draw()
		fig.savefig("plots/tsp/" + title + ".png")
		plt.pause(0.01)
		return fig


class Interface():

	def __init__(self, config_file='config/TSP_config.txt'):
		self.load_config(filename=config_file)
		self.data_manager = DataManager(self.problem_number)

		network = Network(epochs=self.epochs, learning_rate=self.learning_rate,
		                  learning_decay=self.learning_decay, initial_neighborhood=self.initial_neighbourhood,
		                  neighborhood_decay=self.neighbourhood_decay, data_manager=self.data_manager)
		display = Display(network)
		fig = None
		while (network.epochs > 0):
			network.train()
			fig = display.plot_output_weights(fig)
			input_text = "abc"
			while (input_text != "STOP"):
				input_text = input("How many more epochs do you want to train? 0 to quit. ")
				try:
					network.epochs = (int)(input_text)
					input_text = "STOP"
				except:
					print("Fail. Enter a number")

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


if __name__ == '__main__':
	keep_going = True
	while (keep_going):
		config_file_name = input("What config file do you want to run? 'K' to run default. 'Q' to quit. ")
		if (config_file_name.lower() == 'q'):
			break
		if (config_file_name.lower() == 'k'):
			Interface()
		else:
			Interface(config_file=config_file_name)
