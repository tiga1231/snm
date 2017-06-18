import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob

def getOffset(X, chrIndex=1):
    chrNames1 = natsorted(set(X[:,3]))
    chrNames2 = natsorted(set(X[:,15]))

    maxLoc1 = [np.max(X[X[:,3]==c][:,4].astype(np.int)) for c in chrNames1]
    maxLoc2 = [np.max(X[X[:,15]==c][:,16].astype(np.int)) for c in chrNames2]
    if chrIndex == 1:
        res = {chrName: sum(maxLoc1[:chrNames1.index(chrName)]) for chrName in chrNames1}
        res['sum'] = sum(maxLoc1)
    else:
        res = {chrName: sum(maxLoc2[:chrNames2.index(chrName)]) for chrName in chrNames2}
        res['sum'] = sum(maxLoc2)
    return res

#gids = ['11691', '25571', '28814']
#tags = ['chimp', 'human', 'gorilla']
#gids += ['7057', ]#'8143', '28041']
#tags += ['dog', ]#'horse', 'cat']

#plasmodiums
with open('data/plasmodiumTags.txt') as f:
    l = [i.replace('\n','').split(',') for i in f]
    gids = [i[1] for i in l][:3]
    tags = [i[0] for i in l][:3]
print gids
print tags
side = len(gids)
plt.figure(figsize=[6,6])
for i, gid1 in enumerate(gids):
    for j, gid2 in enumerate(gids):
        sub = i*side+j+1 #subplot index
        sub = (side-1-i)*side+j+1 #subplot index
        
        fn = glob('data/ks/plasmodium/' + gid1+'_'+gid2+'*.ks')
        fn += glob('data/ks/' + gid1+'_'+gid2+'*.ks')
        if len(fn) == 0:
            plt.subplot(side,side,sub)
            
            if i==j:
                '''
                plt.text(0,0,tags[i],
                        fontsize=20,
                        horizontalalignment='center',
                        verticalalignment='center')
                '''
                plt.xticks([])
                plt.yticks([])
                plt.axis('square')
                if sub > side*(side-1):
                    plt.xlabel(tags[ (sub-1)%side ])
                if (sub-1) % side == 0:
                    plt.ylabel(tags[ side - sub/side - 1])
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

        offset1 = getOffset(X, 1)
        offset2 = getOffset(X, 2)

        ks = X[:,0].astype(np.float)
        ks[ks==0] = 0.1
        ks = np.log(ks)

        #start location of 2 genomes
        x1 = X[:,4].astype(np.int)
        x2 = X[:,16].astype(np.int)

        x1 += np.array(  [offset1[chrName] for chrName in X[:,3]]  )
        x2 += np.array(  [offset2[chrName] for chrName in X[:,15]]  )

        plt.subplot(side, side, sub)
        plt.scatter(x2, x1, c=ks, s=1, alpha=0.8)
        plt.axis('square')
        if sub > side*(side-1):
            plt.xlabel(tags[ (sub-1)%side ])
        if (sub-1) % side == 0:
            plt.ylabel(tags[ side - sub/side - 1])
        plt.xticks([])
        plt.yticks([])
        #plt.xticks(offset2.values(), offset2.keys())
        #plt.yticks(offset1.values(), offset1.keys())
        lim = max(offset1['sum'],offset2['sum'])
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        plt.grid()
        
        sub = side*j+i+1
        sub = side*(side-j-1)+i+1
        plt.subplot(side,side,sub)
        plt.scatter(x1,x2, c=ks, s=1, alpha=0.8)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        #plt.xticks(offset1.values(), offset1.keys())
        #plt.yticks(offset2.values(), offset2.keys())
        if sub > side*(side-1):
           plt.xlabel(tags[ (sub-1)%side ])
        if (sub-1) % side == 0:
            plt.ylabel(tags[ side - sub/side - 1])
        lim = max(offset1['sum'],offset2['sum'])
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        plt.grid()

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)
plt.show()
