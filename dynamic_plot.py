import matplotlib.pyplot as plt
import os


plt.figure()

def plot_map(cities, nodes, iteration, filenumber, fig=None):
    plt.ion()
    plt.delaxes()
    #fig = fig if fig else plt.figure()
    #axes = fig.gca()

    plt.scatter(*zip(*cities), color='red', s=3)
    plt.scatter(*zip(*nodes), color='green', s=2)

    plt.plot(*zip(*(nodes+[nodes[0]])), color='darkgreen')

    #plt.gca().invert_xaxis()
    plt.gca().set_aspect('equal', adjustable='datalim')

    plt.title('Iteration #{:06d}'.format(iteration))
    #plt.axis('off')
    directory = 'plots/tsp/iterations/' + str(filenumber) + '/'
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    plt.savefig(directory + '{}.png'.format(iteration))
    #plt.clf()
    plt.draw()
    plt.pause(0.01)