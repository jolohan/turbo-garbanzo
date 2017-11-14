import numpy as np
import layer as layer

class Network():

    input_layer = None
    output_layer = None

    def __init__(self, input_activation_func='linear', output_activation_func='linear',
                 input_size=4, output_size=2):
        self.input_layer = layer.Layer(input_activation_func, input_size, output_size)
        self.output_layer = layer.Layer(output_activation_func, input_size, output_size, output_layer=True)

    def run_once(self, input):
        output_from_input_layer = self.input_layer.compute_activation_vector(input)


