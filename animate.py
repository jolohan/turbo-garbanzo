"""
=========================
Simple animation examples
=========================

This example contains two animations. The first is a random walk plot. The
second is an image animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import os

folder = 2
path = 'plots/tsp/iterations/' + str(folder) + "/"

def get_images():
	print("Reading images...")
	
	images = []
	sorted_images = []
	for dirs, subdirs, files in os.walk(path):
		for file in files:
			if ('.png' in file):
				
				sorted_images.append(int(file.split(".")[0]))
	sorted_images = sorted(sorted_images)
	for img_file in sorted_images:
		print(img_file)
		filename = str(img_file) + ".png"
		images.append(mpimg.imread(os.path.join(path, filename)))
				
	print("Done.")
	return images

images = get_images()

# To save the animation, use the command: line_ani.save('lines.mp4')

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

img_plot = []
for img in images:
	img_plot.append([ax.imshow(img)])



im_ani = animation.ArtistAnimation(fig, img_plot, interval=60, repeat_delay=3000,
                                   blit=False)

# To save this second animation with some metadata, use the following command:
# im_ani.save('im.mp4', metadata={'artist':'Guido'})
plt.show()