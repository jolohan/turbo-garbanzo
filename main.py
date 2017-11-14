from network import Network
from data_manager import DataManager
import matplotlib.pyplot as plt
import numpy as np

class Display():

    def __init__(self, network, data_manager):
        weights = network.get_weights()
        weights = np.array(weights)
        x = weights[:,0]
        y = weights[:,1]
        print(weights)
        plt.scatter(x, y, c="blue", alpha=0.5, marker='x', label="Chosen path")

        cities = data_manager.input
        print(cities)
        a = cities[:, 0]
        b = cities[:, 1]
        plt.scatter(a, b, c="red", alpha=0.5, marker='o', label="City")

        plt.xlabel("Leprechauns")
        plt.ylabel("Gold")
        plt.legend(loc=2)
        plt.show()

    def plot_output_weights(self):
        pass

if __name__ == '__main__':
    data_manager = DataManager(0)
    network = Network(data_manager=data_manager)
    network.train()

    display = Display(network, data_manager)
