import numpy as np


class MNIST():
	filename = "MNIST/all_flat_mnist_training_cases_text.txt"

	def __init__(self):
		self.labels, self.images = read_file(self.filename)


def read_file(path):
	with open(path, 'r') as f:
		data = f.read()
		lines = data.split('\n')
		label_list = lines[0]
		labels = [int(i) for i in label_list.split(' ')]
		image_list = lines[1:]
		images = []
		for img in image_list:
			images.append([float(i) for i in img.split(' ')])
		for image in images:
			image = image / np.linalg.norm(image)
	return labels, images
