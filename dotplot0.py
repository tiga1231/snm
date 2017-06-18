import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import json


def getOffset0(X, index=1):
    chrNames1 = natsorted(set(X[:,3]))
    chrNames2 = natsorted(set(X[:,15]))
    if index == 1:
        chrNames = chrNames1
        maxLocations = [np.max(X[X[:,3]==c][:,4].astype(np.int)) for c in chrNames]
    elif index == 2:
        chrNames = chrNames2
        maxLocations = [np.max(X[X[:,15]==c][:,16].astype(np.int)) for c in chrNames]

    res = {chrName: sum(maxLocations[:chrNames.index(chrName)]) for chrName in chrNames}
    res[''] = sum(maxLocations)
    return res

def getOffset(gid):
    with open('data/meta/'+gid) as f:
        chrs = json.load(f)['chromosomes']
        chrs = [  (c['name'],c['length']) for c in chrs   ]
        chrs = natsorted(chrs)
    chrNames = [i[0] for i in chrs]
    maxLocations = [i[1] for i in chrs]
    res = {chrName: sum(maxLocations[:chrNames.index(chrName)]) for chrName in chrNames}
    res[''] = sum(maxLocations)
    return res

fn = 'data/ks/plasmodium/19106_19106.CDS-CDS.last.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks'
fn = 'data/ks/plasmodium/2460_32770.CDS-CDS.last.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks'
gid1, gid2 = fn.split('/')[-1].split('.')[0].split('_')

with open(fn) as f:
    f = (l.replace('||',' ') for l in f)
    X = np.loadtxt(f, dtype = np.str)
    if gid1 == gid2:
        Y = np.copy(X)
        Y[:,3], Y[:,15] = np.copy(Y[:,15]), np.copy(Y[:,3])
        Y[:,4], Y[:,16] = np.copy(Y[:,16]), np.copy(Y[:,4])
        X = np.concatenate([X, Y], axis=0)

#0:ks, larger = more disimilar
#remove not well-defined rows
X = X[X[:,0]!='NA']
X = X[X[:,0]!='undef']

#chrNames1 = natsorted(set(X[:,3]))
#chrNames2 = natsorted(set(X[:,15]))
#offset1 = getOffset(X, index=1)
#offset2 = getOffset(X, index=2)
offset1 = getOffset(gid1)
offset2 = getOffset(gid2)
    
ks = X[:,0].astype(np.float)
#ks = np.exp(-ks)
ks[ks==0] = 0.1
ks = np.log(ks)

#start location of 2 genomes
x1 = X[:,4].astype(np.int)
x2 = X[:,16].astype(np.int)

x1 += np.array(  [offset1[chrName] for chrName in X[:,3]]  )
x2 += np.array(  [offset2[chrName] for chrName in X[:,15]]  )
plt.scatter(x1,x2, c=ks, 
            s=1, #cmap='gray', # darker the more similar (??)
            alpha=0.8)
plt.axis('square')
plt.xticks(offset1.values(), offset1.keys())
plt.yticks(offset2.values(), offset2.keys())
plt.xlim([0, offset1['']])
plt.ylim([0, offset2['']])

plt.xlabel(gid1)
plt.ylabel(gid2)
plt.grid()

plt.show()
