import json
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal as rnorm

def randLoc(center=[0,0], radius = 10):
    r = rnorm(loc=center, scale = radius, size=2)
    return r

def genNode(loc = [0,0], children=None):
    return {
            'x':loc[0],
            'y':loc[1],
            'children':children
            }

def genNodes(loc, var, level=2):
    res = []
    for i in range(4):
        l = randLoc( loc, var)
        res.append(genNode(l, children, level-1))
    return res


dumped = 




with open('level2.js','w') as f:
    f.write('var data = \n')
    json.dump(dumped, f, indent=2)
'''
x = np.array([randLoc([0,0], 20) for i in range(2000)])
plt.scatter(x[:,0], x[:,1])
plt.axis('square')
plt.show()'''
