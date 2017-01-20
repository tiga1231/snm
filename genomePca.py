from glob import glob

import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from kernel import kernel
from clean import cleanKS

import sys

if len(sys.argv)>1 and sys.argv[1]=='xy':
    isXY = True
else:
    isXY = False

np.set_printoptions(precision=4)

#interest = ['11691', '25571','7057','28918','4242','28814','28041','8143']


tags = []
interest = []
with open('ks/tags.txt') as f:
    for line in f:
        line = line.replace('\n','')
        i, t = line.split(',')
        tags.append(t)
        interest.append(i)

print interest
print tags
cleanKS(interest)
path = 'ks/cleaned/*.ks'
files = glob(path)

'''
genomeIDs = [f.split('/')[-1].split('.')[0].split('_') for f in files]
genomeIDs = set(np.array(genomeIDs).ravel())
genomeIDs = [int(g) for g in genomeIDs]
'''

genomeIDs = [int(g) for g in interest]

print genomeIDs

x = pd.DataFrame(columns=genomeIDs, index=genomeIDs, data=0.0)

for f in files:
    print '_'*60
    print f
    genA, genB = f.split('/')[-1].split('.')[0].split('_')
    genA, genB = int(genA), int(genB)
    ki = kernel(f, sigma=1)
    print ki
    x[genA][genB] = ki
    x[genB][genA] = ki

#genomeNames = pd.read_csv('ks/tags.txt',
#                         index_col = 'id')

for i in genomeIDs:
    x[i][i] = 1
    
print x

pca = PCA(n_components=3)
pca.fit(x)
print 'pca ratio', pca.explained_variance_ratio_

xCap = pca.transform(x)
print xCap
if isXY:
    x = [{'x':d[0], 'y':d[1],
        'cat':i,
        'tag':tags[i]} for i,d in enumerate(xCap)]

    with open('data.js', 'w') as f:
        f.write('var data = \n')
        json.dump(x,f, indent = 2)

else:
    fig = plt.figure()
    ax = fig.add_subplot(121, projection = '3d')
    ax.scatter(xCap[:,0],xCap[:,1],xCap[:,2], color = 'bbbbrb')
    for i,pt in enumerate(xCap):
        ax.text(pt[0],pt[1],pt[2],tags[i])
    limit = np.max(np.abs(xCap))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel('1st')
    ax.set_ylabel('2nd')
    ax.set_zlabel('3rd')

    ax = fig.add_subplot(122)
    ax.scatter(xCap[:,0],xCap[:,1], color='bbbbrb')
    ax.axis('square')
    for i,pt in enumerate(xCap):
        print pt, i
        ax.text(pt[0],pt[1], tags[i])
    ##plt.savefig("test.svg")
    plt.show()

