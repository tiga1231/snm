from config import *
import json
from math import exp
import numpy as np


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


def getnpz():
    npz = np.load('data/ks_small.npz')
    ks = npz[npz.keys()[0]]
    print ks[ks<0.1]


if __name__=='__main__':
    getnpz()
    #print kernel2(25571, 11691)
