import matplotlib.pyplot as plt
import os
import math


plt.figure()

def plot(activation_values, nodes, iteration):


    colors = ['lightgreen', 'green', 'magenta', 'blue', 'purple', 'orange', 'red']
    colors.reverse()
    top = max(activation_values)
    top = max(top)
    for i in range(len(nodes)):
        for j in range(len(nodes[i])):
            color = colors[int(float(activation_values[i][j]/top)*(len(colors)-1))]
            size = math.exp(-activation_values[i][j]/top)
            plt.scatter(i, j, color=color, s=10*size)

    plt.title('Iteration #{:06d}'.format(iteration))
    plt.axis('off')
    directory = 'plots/mnist/'
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    plt.savefig(directory + '{}.png'.format(iteration))
    plt.clf()



def plot_winners(windexes, train):
    print("Plotting winning indexes...")

    fig, ax = plt.subplots()
    colors = ['lightgreen', 'pink', 'magenta', 'blue', 'purple', 'orange', 'red', 'yellow', 'indigo', 'azure']
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

