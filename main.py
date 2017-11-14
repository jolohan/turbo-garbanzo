from network import Network
from data_manager import DataManager
import matplotlib.pyplot as plt
import numpy as np

class Display():

    def __init__(self, network, data_manager):
        weights = network.get_weights()
        self.weights = np.array(weights)
        self.cities = data_manager.input


    def plot_output_weights(self):
        x = self.weights[:, 0]
        y = self.weights[:, 1]
        print(x)
        plt.scatter(x, y, c="blue", alpha=0.5, marker='x', label="Chosen path")
        for i in range(len(self.weights)-1):
            x1 = x[i]
            x2 = x[i+1]
            print(x2)
            y1 = y[i]
            y2 = y[i+1]
            plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)
        plt.plot([x[0], x[-1]], [y[0], y[-1]], color='k', linestyle='-', linewidth=1)

        a = self.cities[:, 0]
        b = self.cities[:, 1]
        plt.scatter(a, b, c="red", alpha=0.5, marker='o', label="City")

        plt.xlabel("Leprechauns")
        plt.ylabel("Gold")
        plt.legend(loc=2)
        plt.show()

if __name__ == '__main__':
    data_manager = DataManager(1)

    # Parameters:
    epochs = 200
    learning_rate = 0.01
    learning_decay = 0.5
    initial_neighborhood = 3
    neighborhood_decay = 0.2


    network = Network(epochs=epochs, learning_rate=learning_rate, learning_decay=learning_decay,
        initial_neighborhood=initial_neighborhood, neighborhood_decay=neighborhood_decay, data_manager=data_manager)
    network.train()





    display = Display(network, data_manager)
    display.plot_output_weights()
