from config import *
from kernel import isDataLine

import json
import numpy as np
import matplotlib.pyplot as plt


gid1 = "11691"#chimp
gid2 = "25571"#human
'''
gid1 = "7057"#dog
gid2 = "28041"#cat
'''
try:
    ksf = ksFiles[gid1+'_'+gid2]
except KeyError:
    ksf = ksFiles[gid2+'_'+gid1]
    gid1, gid2 = gid2, gid1

mf1 = metaFiles[gid1]
mf2 = metaFiles[gid2]


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


def offset(name, m):
    names = [i[0] for i in m]
    c = names.index(name)
    return sum([i[1] for i in m][:c])

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

shift1 = np.array([offset(name,m1) for name in chrName1])
shift2 = np.array([offset(name,m2) for name in chrName2])
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

plt.xlabel(genomeTags[gid1])
plt.ylabel(genomeTags[gid2])

x = sorted(list(set(shift1)))
plt.xticks(x, [i[0] for i in m1])
x = sorted(list(set(shift2)))
plt.yticks(x, [i[0] for i in m2])
plt.grid()

plt.colorbar()
#plt.figure()
#plt.hist(X[:,2],bins=75)
plt.show()

