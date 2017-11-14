import numpy as np
from sklearn import preprocessing

class Data_Manager():

    input = np.array()
    dimension = None
    input_size = None
    output_size = None

    def __init__(self, filenumber):
        filename = "TSP Data Euclidean/" + str(filenumber) + '.txt'
        temp_matrix = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            self.dimension = (int)(lines[2].split(' ')[1])
            for i in range(5, len(lines)):
                row = lines[i]
                if (row == 'EOF'):
                    break
                temp_matrix.append(row.split(' ')[1:])

        self.input = np.array(temp_matrix)
        if (self.dimension != len(temp_matrix)):
            print("Something wrong with input loading in def __init__ in Data_Manager")
        self.input_size = self.dimension
        self.output_size = self.dimension*3

    def normalize_data(self, min_value, max_value):
        pass