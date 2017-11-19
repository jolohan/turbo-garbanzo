import numpy as np
import random
import math


class MNIST():
	filename = "MNIST/all_flat_mnist_training_cases_text.txt"

	def __init__(self, size, test_size):
		self.img_labels, self.images = read_file(self.filename)
		to_sort = list(zip(self.images, self.img_labels))
		random.shuffle(to_sort)
		self.images, self.img_labels = zip(*to_sort)
		self.labels = self.img_labels[:int(size)]
		self.input = self.images[:int(size)]
		self.test_labels = self.img_labels[int(size):int(size) + int(test_size)]
		self.test_input = self.images[int(size):int(size) + int(test_size)]
		print("size = " + str(len(self.test_input)))
		self.norm_constant = self.normalize()
		
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
		for i in range(len(self.test_input)):
			for j in range(len(self.test_input[i])):
				self.test_input[i][j] /= norm
		print("Done!")
		return norm

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
