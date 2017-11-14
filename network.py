from layer import Layer
from data_manager import DataManager
import math
import numpy as np
import random

class Network():

    def __init__(self, input_activation_func='euclidean', output_activation_func='euclidean',
                 input_size=2, output_size=100, epochs=10, learning_rate=0.01, learning_decay=1.0):
    	self.data_manager = DataManager(1)
    	self.input_size = self.data_manager.input_size
    	self.output_size = self.data_manager.output_size
    	print(self.input_size, self.output_size)
    	self.input = self.data_manager.input
    	self.epochs = epochs
    	self.similarity = 0.5
    	self.initial_learning_rate = learning_rate
    	self.learning_decay = learning_decay
        self.input_layer = Layer(input_activation_func, self.input_size, self.output_size)
        self.output_layer = Layer(output_activation_func, self.output_size, output_size=0, output_layer=True)

    def forward(self, input):
        output = self.input_layer.compute_activation_vector(input)
       	return output

    def run_once(self, input):
    	return np.argmax(self.forward(input))

    def train(self):
    	for t in range(self.epochs):
    		print("\nTraining Epoch " + str(t))
    		random.shuffle(self.input)
    		for i, train_sample in enumerate(self.input):

    			index = self.run_once(train_sample)
    			winning_neighbors = [(index + 1)%self.output_size, (index-1)%self.output_size]
    			self.optimize_network(index, winning_neighbors, train_sample, t)
    	"""
    	print(self.input[0])
    	forward = self.forward(self.input[0])
    	print(forward)
    	index = self.run_once(self.input[0])
    	print(index)
    	print(self.get_weights_to(index))
    	print(self.get_weights())
    	"""

    def get_weights_to(self, j):
    	return self.input_layer.get_out_weights(j)

    def get_weights(self):
    	return [self.input_layer.get_out_weights(j) for j in range(self.output_size)]


    def optimize_network(self, j, neighbors, input_values, t):
    	learning_rate = self.initial_learning_rate*math.exp(-t/self.learning_decay)
    	for i, n in enumerate(self.input_layer.nodes):
    		n.weights[j] += learning_rate*(input_values[i]-n.weights[j])
    	for k in neighbors:
    		for i, n in enumerate(self.input_layer.nodes):
    			n.weights[k] += learning_rate*self.similarity*(input_values[i]-n.weights[k])


network = Network()
network.train()



