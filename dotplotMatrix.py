import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import json

def getOffset0(X, chrIndex=1):
    chrNames1 = natsorted(set(X[:,3]))
    chrNames2 = natsorted(set(X[:,15]))

    maxLoc1 = [np.max(X[X[:,3]==c][:,4].astype(np.int)) for c in chrNames1]
    maxLoc2 = [np.max(X[X[:,15]==c][:,16].astype(np.int)) for c in chrNames2]
    if chrIndex == 1:
        res = {chrName: sum(maxLoc1[:chrNames1.index(chrName)]) for chrName in chrNames1}
        res[''] = sum(maxLoc1)
    else:
        res = {chrName: sum(maxLoc2[:chrNames2.index(chrName)]) for chrName in chrNames2}
        res[''] = sum(maxLoc2)
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

    
#gids = ['11691', '25571', '28814']
#tags = ['chimp', 'human', 'gorilla']
#gids += ['7057', ]#'8143', '28041']
#tags += ['dog', ]#'horse', 'cat']

#plasmodiums
with open('data/plasmodiumTags.txt') as f:
    l = [i.replace('\n','').split(',') for i in f]
    gids = [i[1] for i in l][:]
    tags = [i[0] for i in l][:]
print gids
print tags
side = len(gids)
plt.figure(figsize=[side*2,side*2])
for i, gid1 in enumerate(gids):
    for j, gid2 in enumerate(gids):
        print i,j       
        sub = i*side+j+1 #subplot index
        sub = (side-1-i)*side+j+1 #subplot index
        fn = glob('data/ks/plasmodium/' + gid1+'_'+gid2+'*.ks')
        fn += glob('data/ks/' + gid1+'_'+gid2+'*.ks')
        if len(fn) == 0: #if no such file
            plt.subplot(side,side,sub)
            continue
        else:
            fn = fn[0]
            with open(fn) as f:
                fl = (l.replace('||',' ') for l in f)
                X = np.loadtxt(fl, dtype = np.str)
                if len(X) == 0:
                    plt.subplot(side,side,sub)
                    plt.plot([0,1],[0,1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                    plt.axis('square')
                    if sub > side*(side-1):
                        plt.xlabel(tags[(sub-1)%side] +'\n'+ gids[(sub-1)%side])
                    if (sub-1) % side == 0:
                        plt.ylabel(tags[side-sub/side-1]+'\n'+gids[side-sub/side-1])
                    continue
                if gid1 == gid2:
                    Y = np.copy(X)
                    Y[:,3], Y[:,15] = np.copy(Y[:,15]), np.copy(Y[:,3])
                    Y[:,4], Y[:,16] = np.copy(Y[:,16]), np.copy(Y[:,4])
                    X = np.concatenate([X, Y], axis=0)
                    
        #0:ks, larger = more disimilar
        #remove undefined rows
        X = X[X[:,0]!='NA']
        X = X[X[:,0]!='undef']

        #offset1 = getOffset0(X, 1)
        #offset2 = getOffset0(X, 2)
	offset1 = getOffset(gid1)
	offset2 = getOffset(gid2)

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
            plt.xlabel(tags[(sub-1)%side] +'\n'+ gids[(sub-1)%side])
        if (sub-1) % side == 0:
            plt.ylabel(tags[side-sub/side-1]+'\n'+gids[side-sub/side-1])
        plt.xticks([])
        plt.yticks([])
        #plt.xticks(offset2.values(), offset2.keys())
        #plt.yticks(offset1.values(), offset1.keys())
        #plt.grid()
        lim = max(offset1[''],offset2[''])
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        #plt.xlim([0, offset1['']])
        #plt.ylim([0, offset2['']])
        if i==j:
            plt.plot([0,lim], [0,lim])
            continue


        #mirror the plot through diag
        sub = side*j+i+1
        sub = side*(side-j-1)+i+1
        plt.subplot(side,side,sub)
        plt.scatter(x1,x2, c=ks, s=1, alpha=0.8)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        #plt.xticks(offset1.values(), offset1.keys())
        #plt.yticks(offset2.values(), offset2.keys())
        #plt.grid()
        if sub > side*(side-1):
            plt.xlabel(tags[(sub-1)%side] +'\n'+ gids[(sub-1)%side])
        if (sub-1) % side == 0:
            plt.ylabel(tags[side-sub/side-1]+'\n'+gids[side-sub/side-1])
        lim = max(offset1[''],offset2[''])
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        #plt.xlim([0, offset1['']])
        #plt.ylim([0, offset2['']])

#plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)
plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=1, wspace=0, hspace=0)

plt.savefig('a.png')
#plt.show()
