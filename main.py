import numpy as np
from sys import argv
from pca import pca
from random import sample


def main():
    
    level = 2
    ids = ['11691','25571','7057','28918','4242','28814','28041','8143']
    ids = sample(ids, 3)
    #pca
    x,tags = pca(level, ids)
    
    #display options
    try:
        mode = argv[1]
    except IndexError:
        mode = 'matplotlib'
        
        
    if mode == 'matplotlib':
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x[:,0],x[:,1],'o')
        ax.axis('equal')
        ax.grid(color='grey', linestyle='-', linewidth=0.3)
        for i,pt in enumerate(x):
            ax.text(pt[0],pt[1], tags[i])
        plt.show()   
        
    elif mode == 'json':
        import json
        data = [{  'x': x[i,0],
                'y': x[i,1],
                'tag': tags[i]
                }
                for i in range(len(ids))]
        with open(argv[2],'w') as f:
            f.write('var data = \n')
            json.dump(data, f, indent=2)

main()
