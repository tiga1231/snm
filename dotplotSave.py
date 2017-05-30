import sys
from time import time
import json
import numpy as np
import matplotlib.pyplot as plt

from config import *
from kernel import isDataLine


t0 = time()
isTimeOn = False
def tick(msg):
    global t0
    if isTimeOn:
        print msg+':', time()-t0
    t0 = time()
   

def length(gid):
    with open(metaFiles[gid]) as f:
        ch = json.load(f)['chromosomes']
        return sum(i['length']for i in ch)
 
 
def main():
    imgs = {}
    for k in ksFiles.keys():
        global t0
        t0 = time()

        ksf = ksFiles[k]
        gid1, gid2 = k.split('_')
        
        print '-' * 20
        print gid1, gid2

        mf1 = metaFiles[gid1]
        mf2 = metaFiles[gid2]


        for mf in [mf1, mf2]:
            with open(mf) as f:
                meta = json.load(f)
                meta = meta['chromosomes']
                meta = [(m['name'],m['length']) for m in meta if m['CDS_count']>0 and m['gene_count']>0 ]

                #[(name, length) tuples]
                meta = sorted(meta,key=lambda x:x[1], reverse=True)
                
                names = [m[0] for m in meta]
                lengths = [m[1] for m in meta]
                offsets = {}
                for n in names:
                    index = names.index(n)
                    offsets[n] = sum(lengths[:index])

            if mf == mf1:
                offset1 = offsets
            elif mf == mf2:
                offset2 = offsets

        tick('1')

        with open(ksf) as f:
            f = (l.replace('||',' ') for l in f)#slow
            X = np.loadtxt(f, dtype = np.str)
        tick('2')

        try:
            #0:ks, larger = more disimilar
            #remove not well-defined rows
            X = X[X[:,0]!='NA']
            X = X[X[:,0]!='undef']
        except IndexError:
            # if no rows at all
            print gid1, gid2, 'skipped'
            continue

        ks = X[:,0].astype(np.float)
        #ks[ks==0] = 0.1
        #ks[ks>2] = 2
        #ks = np.log(ks)
        
        #chromosome name
        chrName1 = X[:,3]
        chrName2 = X[:,15]

        #start location of 2 genomes
        x1 = X[:,4].astype(np.int)
        x2 = X[:,16].astype(np.int)
        
        shift1 = np.array([offset1[name] for name in chrName1])
        shift2 = np.array([offset2[name] for name in chrName2])


        x1 = x1 + shift1
        x2 = x2 + shift2
        
        #scale to image coordinate
        scale = 1e6
        x1 = (x1 / scale).astype(np.int)
        x2 = (x2 / scale).astype(np.int)
        w = int(length(gid1) / scale) + 1
        h = int(length(gid2) / scale) + 1
        
        tick('3')
        img = np.empty([h,w])
        img[:] = np.Inf
        #TODO pick the min ks value for each location
        img[x2,x1] = ks

        
        '''
        print '-'*20
        print 'ks ', np.min(ks), np.max(ks)
        print 'x1 ', np.min(x1), np.max(x1)
        print 'x2 ', np.min(x2), np.max(x2)
        print 'pts', len(x1)
        print 'w,h', w,h
        print 'sum', np.sum(img)
        print 'img', img
        '''

        imgs[gid1+'_'+gid2] = img
        
        tick('4')
   
        '''
        plt.imshow(img, cmap='Greys_r', origin='lower')
        plt.colorbar()
        #plt.savefig(gid1 + '_' + gid2 + '.png')
        #plt.hist(ks, bins = 50)
        plt.show()'''

        #break
    
    print 'saving file...'
    with open('data.npz', 'w') as f:
        np.savez_compressed(f, **imgs)

if __name__ == '__main__':
    main()