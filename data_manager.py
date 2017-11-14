import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize

class DataManager():

    dimension = None
    input_size = None
    output_size = None

    def __init__(self, filenumber):
        filename = "TSP Data Euclidean/" + str(filenumber) + '.txt'
        temp_matrix = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            self.dimension = int(lines[2].split(':')[1])
            for i in range(5, len(lines)):
                row = lines[i]
                if (row.rstrip() == 'EOF'):
                    break
                split_list = [r.rstrip() for r in row.split(' ')[1:3]]
                temp_matrix.append([float(i) for i in split_list])

        self.input = np.array(temp_matrix)
        self.normalize_data()
        print(self.input)
        print("dim: "+str(self.dimension))
        if (self.dimension != len(temp_matrix)):
            print("Something wrong with input loading in def __init__ in Data_Manager")
        self.input_size = 2
        self.output_size = self.dimension*1

    def normalize_data(self, min_value=0, max_value=1):
        self.input[:, 0] = [((x-min(self.input[:, 0]))/(max(self.input[:,0]) - min(self.input[:, 0]))) for x in self.input[:, 0]]
        self.input[:, 1] = [((x-min(self.input[:, 1]))/(max(self.input[:,1]) - min(self.input[:, 1]))) for x in self.input[:, 1]]

