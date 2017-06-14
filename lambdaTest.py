import numpy as np
from sys import argv
from pca import pca
from config import *
import json
import matplotlib.pyplot as plt

def main():
    level = 2
    ids = ['28814', '11691', '28041', '7057', '25571', '8143', '28918', '4242']
    color = ['C' + str(i) for i in range(11)]
    fig = plt.figure()
    for j,lam in enumerate([1e-2, 1e-1, 1, 2, 4, 8, 16, 64, 256]):
        x, tags, m = pca(level, ids, lam)
        ax = fig.add_subplot(3,3,j+1)

        #scatter plot
        '''
        for i, pt in enumerate(x):
            ax.scatter(pt[0], pt[1], s=80, c=color[i])

            if tags[i] in ['human','mouse', 'cat']:
                ax.text(pt[0], pt[1], tags[i],
                horizontalalignment='right')
            else:
                ax.text(pt[0], pt[1], tags[i])
        plt.title('lambda = '+str(lam))
        ax.axis('equal')
        ax.grid(color='grey', linestyle='-', linewidth=0.3)
        '''

        #matrix plot
        plt.title('lambda = '+str(lam))
        ax.imshow(m)
    plt.show()   
 
if __name__ == '__main__':   
    main()
