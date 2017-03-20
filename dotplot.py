from config import *
from kernel import isDataLine

import json
import numpy as np
import matplotlib.pyplot as plt

ksf = ksFiles['11691_25571']
mf1 = metaFiles['11691']#chimp
mf2 = metaFiles['25571']#human


for mf in [mf1, mf2]:
    with open(mf) as f:
        meta = json.load(f)
        meta = meta['chromosomes']
        meta = [(i['name'],i['length']) for i in meta]
        meta = sorted(meta,key=lambda x:x[1], reverse=True)
    if mf == mf1:
        m1 = meta
    elif mf == mf2:
        m2 = meta


def offset1(name):
    names = [i[0] for i in m1]
    c = names.index(name)
    return sum([i[1] for i in m1][:c])

def offset2(name):
    names = [i[0] for i in m2]
    c = names.index(name)
    return sum([i[1] for i in m2][:c])

def getMin(name, x1, names):
    return np.min([x1[i] for i in xrange(len(x1)) if names[i]==name])


with open(ksf) as f:
    f = (l.replace('||',' ') for l in f)
    X = np.loadtxt(f, dtype = np.str)

X = X[X[:,0]!='NA']
#X = X[X[:,0].astype(np.float)<0.1]

ks = X[:,0].astype(np.float)
ks[ks==0] = 0.1
ks = np.log(ks)
chrName1 = X[:,3]
chrName2 = X[:,15]
x1 = X[:,4].astype(np.int)
x2 = X[:,16].astype(np.int)

shift1 = np.array([offset1(name) for name in chrName1])
shift2 = np.array([offset2(name) for name in chrName2])
'''
mins1 = dict([name, getMin(name, x1, chrName1)] for name in set(chrName1))
mins2 = dict([name, getMin(name, x2, chrName2)] for name in set(chrName2))
o1 = np.array([mins1[name] for name in chrName1])
o2 = np.array([mins2[name] for name in chrName2])
'''
x1 = x1 + shift1# - o1
x2 = x2 + shift2# - o2

plt.scatter(x1,x2, c=ks, 
            s=5, cmap='rainbow',
            alpha=0.8)
plt.xticks(list(set(shift1)))
plt.yticks(list(set(shift2)))
plt.grid()

plt.colorbar()
#plt.figure()
#plt.hist(X[:,2],bins=75)
plt.show()

