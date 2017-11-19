import matplotlib.pyplot as plt
import os
import math


plt.figure()

def plot_winners(windexes, train):
    print("Plotting winning indexes...")

    fig2, ax = plt.subplots()
    colors = ['lightgreen', 'pink', 'magenta', 'blue', 'purple', 'orange', 'red', 'yellow', 'indigo', 'black']
    for index, label in windexes:
        color = colors[int(label)]
        size = 20
        ax.scatter(index[0], index[1], color=color, s=size, alpha=0.5)
        ax.annotate(str(label), (index[0], index[1]))

    if (train):
        plt.title("Training Set Test")
    else:
        plt.title('Testing Set Test')

    directory = 'plots/mnist/'
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    if (train):
        filename = "training_test"
    else:
        filename = "testing_test"
    plt.savefig(directory + filename + '.png')
    plt.clf()

def plot_labels(nodes, iteration):

    plt.ion()
    plt.delaxes()
    colors = ['green', 'pink', 'magenta', 'blue', 'purple', 'orange', 'red', 'yellow', 'indigo', 'black']
    for i in range(len(nodes)):
        for j in range(len(nodes[i])):
            lab = int(nodes[i][j].label - 1)
            if (lab >= 0):
                color = colors[lab]
                marker = 'o'
            else:
                color = 'black'
                marker = 'x'
            size = 20
            plt.scatter(i, j, color=color, s=size, marker=marker, alpha=0.9)
            #ax.annotate(str(lab), (i, j))

    plt.title('Iteration #{:06d}'.format(iteration))
    directory = 'plots/mnist/'
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    plt.savefig(directory + '{}.png'.format(iteration))
    plt.draw()
    plt.pause(0.01)
