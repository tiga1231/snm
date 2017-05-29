import sys

from config import *
from kernel import isDataLine

import json
import numpy as np
import matplotlib.pyplot as plt


def offset(name, meta):
    names = [i[0] for i in meta]
    c = names.index(name)
    return sum( [i[1] for i in meta] [:c] )

def length(gid):
    with open(metaFiles[gid]) as f:
        ch = json.load(f)['chromosomes']
        return sum(i['length']for i in ch)
        
'''
def getMin(name, x1, names):
    return np.min([x1[i] for i in xrange(len(x1)) if names[i]==name])
'''
def main():
    for k in ['28814_28918',]:#ksFiles.keys():
        ksf = ksFiles[k]
        gid1, gid2 = k.split('_')
        
        print gid1, gid2

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

        with open(ksf) as f:
            f = (l.replace('||',' ') for l in f)
            X = np.loadtxt(f, dtype = np.str)

        #0:ks, larger = more disimilar
        #remove not well-defined rows
        try:
            X = X[X[:,0]!='NA']
            X = X[X[:,0]!='undef']
        except IndexError:
            print gid1, gid2, 'skipped'
            continue
        ks = X[:,0].astype(np.float)
        ks[ks==0] = 0.1
        ks = np.log(ks)

        #chromosome name
        chrName1 = X[:,3]
        chrName2 = X[:,15]

        #start location of 2 genomes
        x1 = X[:,4].astype(np.int)
        x2 = X[:,16].astype(np.int)

        shift1 = np.array([offset(name,m1) for name in chrName1])
        shift2 = np.array([offset(name,m2) for name in chrName2])

        x1 = x1 + shift1
        x2 = x2 + shift2


        mode = 'save'
        if mode == 'display':
            plt.scatter(x1,x2, c=ks, 
                        s=1, cmap='gray', # darker the more similar
                        alpha=0.8)
            plt.axis('square')
            plt.xlabel(genomeTags[gid1])
            plt.ylabel(genomeTags[gid2])
            x = sorted(list(set(shift1)))
            plt.xticks(x, [i[0] for i in m1])
            x = sorted(list(set(shift2)))
            plt.yticks(x, [i[0] for i in m2])
            plt.grid()
            plt.colorbar()
            plt.show()

        elif mode == 'save':

            w,h = length(gid1), length(gid2)
            print 'w,h = ', w, h 
            figsize = [w*10e-9, h*10e-9]
            print 'figsize', figsize
            fig = plt.figure(figsize = figsize)
            ax=fig.add_axes((0, 0, 1, 1))
            ax.axis('square')
            scatter = ax.scatter(x1,x2, c=ks, 
                        s=1, cmap='gray', # darker the more similar
                        alpha=0.8)
            ax.set_xlim([0,np.max(x1)])
            ax.set_ylim([0,np.max(x2)])
            #plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            plt.savefig(gid1 + '_' + gid2 + '.png')

if __name__ == '__main__':
    main()
    '''
    with open('./data/meta/7057') as f:
	meta = json.load(f)
	meta = meta['chromosomes']
	meta = [(i['name'],i['length']) for i in meta]
	meta = sorted(meta,key=lambda x:x[1], reverse=True)
    print offset('1',meta)
    '''
