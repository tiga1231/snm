import numpy as np
from sys import argv
from pca import pca
from config import *
import json

def main():
    level = 2
    #ids = ['11691','25571','7057','28918','4242','28814','28041','8143']    
    ids = ['28814','8143','7057','11691','28041','25571']
    x,tags,m = pca(level, ids)
    
    #display options
    try:
        mode = argv[1]
    except IndexError:
        mode = 'matplotlib'
        
        
    if mode == 'matplotlib':
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, pt in enumerate(x):
            ax.scatter(pt[0], pt[1], s=80)
            if tags[i] in ['human','dog']:
                ax.text(pt[0], pt[1], tags[i],
                fontsize=14,
                horizontalalignment='right')
            else:
                ax.text(pt[0], pt[1], tags[i],
                fontsize=14)
        ax.axis('equal')
        ax.grid(color='grey', linestyle='-', linewidth=0.3)

        '''
        plt.subplot(122)
        plt.imshow(m)
        plt.colorbar()
        plt.xticks(range(len(tags)), tags)
        plt.yticks(range(len(tags)), tags)
        '''
        plt.show()   
        
    elif mode == 'json':
        data = [{  'x': x[i,0],
                'y': x[i,1],
                'tag': tags[i]
                }
                for i in range(len(ids))]
        with open(argv[2],'w') as f:
            f.write('var data = \n')
            json.dump(data, f, indent=2)

main()
