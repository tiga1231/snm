from glob import glob
import sys
import json

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kernel import kernel,centralize
from clean import cleanKS


dataRoot = 'data/'
debug = True

if len(sys.argv)>1:
    isJson = True
    outFile = sys.argv[1]
else:
    isJson = False

np.set_printoptions(precision=2)

includeGenome = ['11691', '25571','7057','28918','4242','28814','28041','8143']

tags = []
interest = []
with open(dataRoot + 'tags.txt') as f:
    for line in f:
        line = line.replace('\n','')
        i, tag = line.split(',')
        if i in includeGenome:
            interest.append(i)
            tags.append(tag)

if debug:
    print 'tags'
    print tags

cleanKS(interest)
path = dataRoot + 'cleaned/*.ks'
files = glob(path)

genomeIDs = [int(g) for g in interest]

if debug:
    print 'genomeIDs'
    print genomeIDs


x = np.ndarray(shape=[len(interest), len(interest)] )

for f in files:
    genA, genB = f.split('/')[-1].split('.')[0].split('_')
    genA, genB = int(genA), int(genB)
    i,j = genomeIDs.index(genA), genomeIDs.index(genB)
    ki = kernel(f, sigma=1)
    x[i][j] = ki
    x[j][i] = ki

for i in range(len(genomeIDs)):
    x[i][i] = 1

if debug:
    print 'dot product matrix'
    print x

x = centralize(x)

pca = PCA(n_components=3)
pca.fit(x)
if debug:
    print 'pca ratio', pca.explained_variance_ratio_


xCap = pca.transform(x)

if isJson:
    x = [{'x':d[0], 'y':d[1],
        'cat':i,
        'tag':tags[i]} for i,d in enumerate(xCap)]

    with open(outFile, 'w') as f:
        f.write('var data = \n')
        json.dump(x,f, indent = 2)

else:
    fig = plt.figure()
    '''
    #3d plot on first 3 pc
    ax = fig.add_subplot(121, projection = '3d')
    ax.scatter(xCap[:,0],xCap[:,1],xCap[:,2])
    for i,pt in enumerate(xCap):
        ax.text(pt[0],pt[1],pt[2],tags[i])
    limit = np.max(np.abs(xCap))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel('1st')
    ax.set_ylabel('2nd')
    ax.set_zlabel('3rd')
    '''
    ax = fig.add_subplot(111)
    ax.scatter(xCap[:,0],xCap[:,1])
    ax.axis('square')
    for i,pt in enumerate(xCap):
        ax.text(pt[0],pt[1], tags[i])
    ##plt.savefig("test.svg")
    plt.show()

