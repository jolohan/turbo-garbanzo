from layer import Layer
from data_manager import DataManager
import math


class Network():

    def __init__(self, input_activation_func='euclidean', output_activation_func='euclidean',
                 input_size=4, output_size=2, epochs=10, learning_rate=0.01, learning_decay=1.0):
    	self.data_manager = DataManager()
    	self.input_size = self.data_manager.input_size
    	self.output_size = self.data_manager.output_size
    	self.input = self.data_manager.input
    	self.epochs = epochs
    	self.similarity = 0.5
    	self.initial_learning_rate = learning_rate
    	self.learning_decay = learning_decay
        self.input_layer = Layer(input_activation_func, input_size, output_size)
        self.output_layer = Layer(output_activation_func, input_size, output_size, output_layer=True)

    def forward(self, input):
        output = self.input_layer.compute_activation_vector(input)
       	return output

    def run_once(self, input):
    	return np.argmax(forward(input))

    def train(self):
    	for t in range(self.epochs):
    		print("\nTraining Epoch " + str(t))
    		for i, train_sample in enumerate(self.input):

    			index = run_once(train_sample)
    			winning_neighbors = [(index + 1)%self.output_size, (index-1)%self.output_size]
    			self.optimize_network(index, winning_neighbors, train_sample, t)

    def optimize_network(self, j, neighbors, input_values, t):
    	learning_rate = self.initial_learning_rate*math.exp(-t/self.learning_decay)
    	for i, w in enumerate(input_layer.nodes[j].weights):
    		w += learning_rate*(input_values[i]-w[j][i])
    	for k in neighbors:
    		for i, w in enumerate(input_layer.nodes[k].weights):
    			w += learning_rate*self.similarity*(input_values[i]-w[k][i])


network = Network()
network.train()



