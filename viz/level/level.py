import json
from random import random, randint

nodeId = -1

def genNode(x,y,level,parent):
    global nodeId
    nodeId += 1
    return {'id':nodeId,
            'level':level,
            'x':x,
            'y':y,
            'parent':parent,
            }

def r(c,v):
    return randint(c-v,c+v)

data = [genNode( (1+i)*500/3,(1+j)*500/3,
                 level=0,
                 parent = (i,j))
        for i in range(2)
        for j in range(2)
        ]

temp = []
for node in data:
    temp += [genNode( r(node['x'],50),r(node['y'],60),
                      level=1,
                      parent = node['id'])
            for i in range(3)
            ]
    
temp2 = []
for node in temp:
    temp2 += [genNode( r(node['x'],30),r(node['y'],40),
                      level=2,
                      parent = node['id'])
            for i in range(5)
            ]
data+=temp+temp2

    
with open('data.js','w') as f:
    f.write('var data = \n')
    f.write(json.dumps(data))
