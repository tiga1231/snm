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
        res['sum'] = sum(maxLocations1)
    else:
        res = {chrName: sum(maxLocations2[:chrNames2.index(chrName)]) for chrName in chrNames2}
        res['sum'] = sum(maxLocations2)
    return res

gids = ['11691', '25571', '28814']
tags = ['chimp', 'human', 'gorilla']
#gids += ['7057', '8143', '28041']
#tags += ['dog', 'horse', 'cat']
side = len(gids)
plt.figure(figsize=[6,6])
for i, gid1 in enumerate(gids):
    for j, gid2 in enumerate(gids):
        sub = i*side+j+1
        
        fn = glob('data/ks/' + gid1+'_'+gid2+'*.ks')
        if len(fn) == 0:
            plt.subplot(side,side,sub)
            if i==j:
                plt.text(0,0,tags[i],
                        fontsize=20,
                        horizontalalignment='center',
                        verticalalignment='center')
            plt.xticks([])
            plt.yticks([])
            plt.axis('square')
            continue
        else:
            fn = fn[0]
        with open(fn) as f:
            fl = (l.replace('||',' ') for l in f)
            print fl
            X = np.loadtxt(fl, dtype = np.str)

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

        plt.subplot(side, side, sub)
        plt.scatter(x1,x2, c=ks, 
                    s=1, #cmap='gray', # darker the more similar (??)
                    alpha=0.8)
        plt.axis('square')
        #plt.xticks(offset1.values(), offset1.keys())
        #plt.yticks(offset2.values(), offset2.keys())
        plt.xticks([])
        plt.yticks([])
        lim = max(offset1['sum'],offset2['sum'])
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        
        sub2 = side*j+i+1
        plt.subplot(side,side,sub2)
        plt.scatter(x2,x1, c=ks, 
                    s=1, #cmap='gray', # darker the more similar (??)
                    alpha=0.8)
        plt.axis('square')
        #plt.xticks(offset1.values(), offset1.keys())
        #plt.yticks(offset2.values(), offset2.keys())
        plt.xticks([])
        plt.yticks([])
        lim = max(offset1['sum'],offset2['sum'])
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        #plt.grid()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

plt.show()
