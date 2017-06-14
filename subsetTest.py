import numpy as np
from sys import argv
from pca import pca
from config import *
import json
import matplotlib.pyplot as plt

def main():
    level = 2
    
    ids = ['28814', '11691', '28041', '7057', '25571', '8143', '28918', '4242']
    
    fig = plt.figure()
    color = ['C'+str(i) for i in range(10)]
    for j in range(6):
        if j==0:
            idi = ids[:]
        else:
            idi = ids[:-j]
        print idi
        x,tags = pca(level, idi)
        ax = fig.add_subplot(2,3,j+1)
        for i, pt in enumerate(x):
            ax.scatter(pt[0], pt[1], c=color[i])
            if tags[i] in ['cat', 'chimp', 'human']:
                ax.text(pt[0], pt[1], tags[i], horizontalalignment='right', fontsize=13)
            else:
                ax.text(pt[0], pt[1], tags[i], fontsize=13)
        ax.axis('equal')
        if j==0:
            plt.title('all')
        else:
            plt.title('excluded ' + genomeTags[ids[-j]])
        ax.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()   
        

main()
