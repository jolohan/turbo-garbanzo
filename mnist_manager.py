import numpy as np
import random
import math


class MNIST():
	filename = "MNIST/all_flat_mnist_training_cases_text.txt"

	def __init__(self, size, test_size):
		self.labels, self.input = read_file(self.filename)
		self.labels = self.labels[:int(size)]
		self.input = self.input[:int(size)]
		self.test_labels = self.labels[int(size):int(test_size)]
		self.test_input = self.input[int(size):int(test_size)]
		self.norm_constant, self.input = self.normalize()

		to_sort = list(zip(self.input, self.labels))
		random.shuffle(to_sort)
		self.input, self.labels = zip(*to_sort)
		self.output_size = int(math.sqrt(len(self.input[0])))
		self.input_size = len(self.input[0])
		self.file = "MNIST"


	def normalize(self):
		max_img = 255.0
		min_img = 0.0
		norm = max_img
		print("Normalizing images with constant " + str(norm))
		for i in range(len(self.input)):
			for j in range(len(self.input[i])):
				self.input[i][j] /= norm
		print("Done!")
		return norm, self.input

def read_file(path):
	print("Reading MNIST examples...")
	with open(path, 'r') as f:
		data = f.read()
		lines = data.split('\n')
		label_list = lines[0]
		labels = [int(i) for i in label_list.split(' ')]
		image_list = lines[1:]
		images = []
		for img in image_list:
			images.append([float(i) for i in img.split(' ')])

	return labels, images
