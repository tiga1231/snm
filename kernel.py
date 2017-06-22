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
                
                
def genomeKernelMatrix(ids, lam=2):
    n = len(ids)
    x = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            k = genomeKernel(ids[i], ids[j], lam)
            x[i,j] = k
            x[j,i] = k
    return x


def genomeKernel(a,b, lam=2):
    '''genome level comparison'''
    a,b = str(a), str(b)
    print a,b
    if a==b:
        return 1
    elif a+'_'+b in ksFiles:
        fn = ksFiles[a+'_'+b]
    elif b+'_'+a in ksFiles:
        fn = ksFiles[b+'_'+a]
    res = 0
    with open(fn) as f:
        for line in f:
            if isDataLine(line):
                x = float(line.split('\t')[0])
                res += exp(-x * lam)

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


def filterTest(lam=1.0, mode='all'):
    '''compute similarity w/ or w/o diag and dump a dict to .pkl'''
    import matplotlib.pyplot as plt
    npz = np.load('data/ks_small.npz')
    kk = {}
    for name,ks in npz.items():
        a,b = name.split('_')
        #print '-'*20
        print name
        #print '-'*20
        ks = np.exp(- lam * ks)

        #TODO the normalizing factor is to be discussed
        #k = np.sum(ks) / (countGene(a) * countGene(b))**0.5
        #ka = np.sum(diagFilter(ks)) / (ks.shape[0]*ks.shape[1])**0.5
        #kb =  np.sum(diagFilter(ks)) / (countGene(a) * countGene(b))**0.5
        
        if mode == 'all':
            #4 is the max ks sum across rows
            k = np.sum(ks) /4 /max(ks.shape[0], ks.shape[1])
        elif mode == 'diag':
            k = np.sum(diagFilter(ks)) /3.3 /max(ks.shape[0], ks.shape[1])
        elif mode == 'anti':
            k = np.sum(diagFilter(ks, True)) /3.3 /max(ks.shape[0], ks.shape[1])
        kk[(a,b)] = k
    print 'dumping data/ks.pkl'
    with open('data/ks.pkl','wb') as f:
        pickle.dump(kk, f)


def pcaTest(exclude=[]):
    from sklearn.decomposition import KernelPCA
    import matplotlib.pyplot as plt
    from config import genomeTags

    with open('data/ks.pkl','rb') as f:
        k = pickle.load(f)
    gids = list(set([i[0] for i in k.keys()] + [i[1] for i in k.keys()]))
    gids = ['2460', '32770', '35091', '35092', '32801', 
            '19106', '32826', '32958', '32902', '35095', 
            '35093', '32903', '32865', '19306', '32904', 
            '32788', '35088'
            ]
    tags = np.array([genomeTags[i] for i in gids])
    print gids
    print tags
    m = np.eye(len(gids), len(gids))
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
    #sort matrix by similarity
    '''
    for i in range(len(m[0])-1):
        order = sorted( zip(m[i:, i:][0], range(len(m[0])-i)) , reverse=True)
        order =  [o[1] for o in order]
        m[i:,i:] = m[i:,i:][order]
        m[i:,i:] = m[i:,i:][:,order]
        tags[i:] = tags[i:][order] 
    '''
    np.set_printoptions(precision=2)
    print m
    plt.subplot(121)
    plt.imshow(m, origin='lower')
    
    plt.xticks(range(len(gids)),tags, rotation=90)
    plt.yticks(range(len(gids)),tags)
    pca = KernelPCA(kernel='precomputed',
                    n_components=3)

    x = pca.fit_transform(m)

    print x
    for i in x:
        print i[0]
    print '-'*20
    for i in x:
        print i[1]
    print '-'*20
    for i in x:
        print i[2]
    for i in tags:
        print i.split('.')[-1]
    plt.subplot(122)
    color = ['#a6cee3'] * 8
    color += ['#1f78b4'] * 5
    color += ['#b2df8a'] * 3
    color += ['#33a02c'] * 1
    for i,g in enumerate(gids):
        plt.scatter(x[i,0], x[i,1], c=color[i])
        if i%2==0:
            plt.text(x[i,0], x[i,1],genomeTags[g], horizontalalignment='right')
        else:
            plt.text(x[i,0], x[i,1],genomeTags[g])
    plt.axis('equal')


def lambdaTest():
    import matplotlib.pyplot as plt
    for i,lam in enumerate([1,]):
        filterTest(lam)
        plt.subplot(1,1,i+1)
        plt.axis('equal')
        plt.grid(color='grey', linestyle='-', linewidth=0.3)
        plt.title('diag only, $e^{-%.1f*ks}$' % lam)
        pcaTest()
    plt.show()

def subsetTest():
    filterTest(lam=32.0)
    excludes = [[ ],
                ['4242',], 
                ['4242','28918'], 
                ['4242','28918', '7057'], 
                ['4242','28918', '7057', '8143'], 
                ['4242','28918', '7057', '8143', '28041'], 
                ]
    for i,e in enumerate(excludes):
        plt.subplot(2,3,i+1)
        pcaTest(e)
        plt.grid(color='grey', linestyle='-', linewidth=0.3)
    plt.show()
    

if __name__=='__main__':
    import matplotlib.pyplot as plt
    for mode in ['all', 'diag', 'anti']:
        plt.figure()
        filterTest(1.0, mode)
        pcaTest()
    plt.show()
