import numpy as np
from sys import argv
from pca import pca
from config import *
import json

def main():
    level = 2
    ids = ['11691','25571','7057','28918','4242','28814','28041','8143']    
    x,tags = pca(level, ids)
    sizes = []
    for i in ids:
        with open(metaFiles[i]) as f:
            meta = json.load(f)
            size = reduce(lambda a,x:a+x['length'], meta['chromosomes'], 0)
            sizes.append(size)
    print sizes
    
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
            s = 100.0 * sizes[i]/ max(sizes)
            ax.scatter(pt[0], pt[1], s=s)
            ax.text(pt[0], pt[1], tags[i])
        ax.axis('equal')
        ax.grid(color='grey', linestyle='-', linewidth=0.3)
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
