import matplotlib.pyplot as plt
import numpy as np

from data_manager import DataManager
from network import Network


class Display():
	def __init__(self, network, data_manager):
		self.network = network
		self.data_manager = data_manager
		plt.ion()

	def plot_output_weights(self, fig=None, title='tits'):
		weights = self.network.get_weights()
		weights = np.array(weights)
		cities = self.data_manager.input
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

class ConfigFileLoader:

	def load_config(self, filename='config/TSP_config.txt'):

		# Pre-processing config file:
		network_dict = {}
		with open(filename, "r") as file:
			for line in file:
				listed_specs = line.split(" ")
				network_dict[listed_specs[0]] = [item.strip() for item in listed_specs[1:]]

		# Parameters for output generation:
		self.vint = network_dict['VInt'][0]
		if (self.vint == "None" or self.vint == "0"):
			self.vint = None
		else:
			self.vint = int(self.vint)

		self.showint = network_dict['ShowInt'][0]
		if (self.showint == "None" or self.showint == "0"):
			self.showint = None
		else:
			self.showint = int(self.showint)

		# 1. Network Dimensions (+ sizes)
		sizes = network_dict['NetSize']
		self.sizes = []

		for s in sizes:
			self.sizes.append(int(s))

if __name__ == '__main__':
	configloader = ConfigFileLoader()
	data_manager = DataManager(0)
	network = Network(data_manager=data_manager)

	# Parameters:
	epochs = 100
	learning_rate = 0.1
	learning_decay = 500.0
	initial_neighborhood = 15
	neighborhood_decay = 200.0

	network = Network(epochs=epochs, learning_rate=learning_rate,
	                  learning_decay=learning_decay, initial_neighborhood=initial_neighborhood,
	                  neighborhood_decay=neighborhood_decay, data_manager=data_manager)
	display = Display(network, data_manager)
	fig = None
	while (network.epochs>0):
		network.train()
		fig = display.plot_output_weights(fig)
		input_text = "abc"
		while(input_text != "STOP"):
			input_text = input("How many more epochs do you want to train? 0 to quit. ")
			try:
				network.epochs = (int)(input_text)
				input_text = "STOP"
			except:
				print("Fail. Enter a number")
