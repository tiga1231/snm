import json
from math import exp
import pickle

import numpy as np
from scipy.signal import convolve2d

from config import *

'''Kernel levels:
1- species      2- genome
3- chromosome   4- gene
'''

def isDataLine(line):
    return not (    line.startswith('#')
                    or line.startswith('NA')
                    or line.startswith('undef')
                )
                
                
def genomeKernelMatrix(ids):
    n = len(ids)
    x = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            k = genomeKernel(ids[i], ids[j])
            x[i,j] = k
            x[j,i] = k
    return x


def genomeKernel(a,b):
    '''genome level comparison'''
    a,b = str(a), str(b)
    if a==b:
        return 1
    elif a+'_'+b in ksFiles:
        fn = ksFiles[a+'_'+b]
    elif b+'_'+a in ksFiles:
        fn = ksFiles[b+'_'+a]
    res = 0
    with open(fn) as f:
        for line in f:
            if not (line.startswith('#')
                    or line.startswith('NA')
                    or line.startswith('undef')):
                x = float(line.split('\t')[0])
                res += exp(-x * 2)

    geneCount = countGene(a) * countGene(b)
    res = res / geneCount**0.5
    return res


def countGene(gid):
    ''' count number of genes in a genome,
    from mata data file'''
    fn = metaFiles[gid]
    with open(fn) as f:
        data = json.loads(f.read())
    res = sum( chrom['gene_count'] for chrom in data['chromosomes'] )
    return res


def countChromosome(gid):
    fn = metaFiles[gid]
    with open(fn) as f:
        data = json.loads(f.read())
    res = len(data['chromosomes'])
    return res
    
'''
def getOffset(ids):
    off = [0]
    for i in ids:
        a = countGene2(i)
        off.append(off[-1]+a)
    return off
'''

def geneMatrix(a,b):
    m = countGene(a)
    n = countGene(b)
    x = np.zeros([m,n])
    #TODO
    return x


def chromosomeKernel(a,b):
    #TODO
    #kernel matrix of chromosomes of all genomes in ids
    x = np.eye(8)
    tags = [str(i) for i in range(x.shape[0])]
    return x,tags


def chromosomeKernelMatrix(ids):
    #TODO
    pass



def diagFilter(img, anti=False):

    '''
    The matrix starts at upper left corner
    so the diag mask is designed to capture elements of form
    img_{i,i}, img_{i,i+1}, img_{i, i+k}... 
    '''

    if anti:
        mask = np.array([[-.5,  0,  0],
                         [  0,  1,  0],
                         [  0,  0,-.5]])
    else:
        mask = np.array([[  0,  0,-.5],
                         [  0,  1,  0],
                         [-.5,  0,  0]])
    result = convolve2d(img,mask,'same')
    result[result<0] = 0
    return result


def filterTest(lam=1.0):
    '''compute similarity w/ or w/o diag and dump a dict to .pkl'''
    import matplotlib.pyplot as plt
    npz = np.load('data/ks_small.npz')
    kk = {}
    for name,ks in npz.items():
        a,b = name.split('_')
        #print '-'*20
        print name,
        #print '-'*20
        ks = np.exp(- lam * ks)
        #TODO the normalizing factor is to be discussed
        #k = np.sum(ks) / (countGene(a) * countGene(b))**0.5
        ka = np.sum(diagFilter(ks)) / (countGene(a) * countGene(b))**0.5
        #kb =  np.sum(diagFilter(ks)) / (countGene(a) * countGene(b))**0.5
        
        #print ka

        #kk[(a,b)] = k
        kk[(a,b)] = ka
        #kk[(a,b)] = kb

    with open('data/ks.pkl','wb') as f:
        pickle.dump(kk, f)
    print 


def pcaTest():
    from sklearn.decomposition import KernelPCA
    import matplotlib.pyplot as plt
    from config import genomeTags

    with open('data/ks.pkl','rb') as f:
        k = pickle.load(f)
    gids = list(set([i[0] for i in k.keys()] + [i[1] for i in k.keys()]))
    
    exclude = ['3068','8']
    for e in exclude:
        gids.remove(e)
    print gids
    print [genomeTags[i] for i in gids]
    m = np.zeros([len(gids), len(gids)])
    for i,x in enumerate(gids):
        for j,y in enumerate(gids):
            if i == j:
                m[i,j] = 1
            elif (x,y) in k:
                m[i,j] = k[(x,y)]
                m[j,i] = k[(x,y)]
            elif (y,x) in k:
                m[i,j] = k[(y,x)]
                m[j,i] = k[(y,x)]
    
    np.set_printoptions(precision=2)
    print m
    pca = KernelPCA(kernel='precomputed',
                    n_components=2)
    x = pca.fit_transform(m)
    for i,g in enumerate(gids):
        plt.scatter(x[i,0], x[i,1])
        plt.text(x[i,0], x[i,1],genomeTags[g])
    #plt.show()

    

if __name__=='__main__':
    #print kernel2(25571, 11691)
    import matplotlib.pyplot as plt
    for i,lam in enumerate([0.1, 1,2,4,8, 16, 32, 64, 128]):
        filterTest(lam)
        plt.subplot(3,3,i+1)
        plt.axis('equal')
        plt.grid(color='grey', linestyle='-', linewidth=0.3)
        plt.title('diag only, $e^{-%.1f*ks}$' % lam)
        pcaTest()
    plt.show()
