import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob

def getOffset(X, index=1):
    chrNames1 = natsorted(set(X[:,3]))
    chrNames2 = natsorted(set(X[:,15]))

    maxLocations1 = [np.max(X[X[:,3]==c][:,4].astype(np.int)) for c in chrNames1]
    maxLocations2 = [np.max(X[X[:,15]==c][:,16].astype(np.int)) for c in chrNames2]
    if index == 1:
        res = {chrName: sum(maxLocations1[:chrNames1.index(chrName)]) for chrName in chrNames1}
        res[''] = sum(maxLocations1)
    else:
        res = {chrName: sum(maxLocations2[:chrNames2.index(chrName)]) for chrName in chrNames2}
        res[''] = sum(maxLocations2)
    return res


##parasite 1
with open('data/ks/plasmodium/2460_32770.CDS-CDS.last.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks') as f:

##parasite 2
#with open('19306_2460.CDS-CDS.last.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks') as f:

##human chimp
#with open('data/ks/11691_25571.CDS-CDS.last.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks') as f:

## 3068 8
#with open('data/ks/3068_8.CDS-CDS.dcmegablast.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks') as f:

## plasmodium
#with open('data/ks/plasmodium/2460_32770.CDS-CDS.last.tdd10.cs0.filtered.dag.all.go_D20_g10_A5.aligncoords.gcoords.ks') as f:
    f = (l.replace('||',' ') for l in f)
    X = np.loadtxt(f, dtype = np.str)

#0:ks, larger = more disimilar
#remove not well-defined rows
X = X[X[:,0]!='NA']
X = X[X[:,0]!='undef']

#chrNames1 = natsorted(set(X[:,3]))
#chrNames2 = natsorted(set(X[:,15]))
offset1 = getOffset(X, index=1)
offset2 = getOffset(X, index=2)

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
#plt.xlabel('Arabidopsis lyrata (v1)')
#plt.ylabel('Arabidopsis thaliana Col-0 (thale cress) (v8)')
plt.grid()

plt.show()
