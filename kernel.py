from config import *
import json
from math import exp
import numpy as np


'''Kernel levels:
1- species      2- genome
3- chromosome   4- gene
'''


def kernel2(a,b):
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

    geneCount = countGene2(a) * countGene2(b)
    res = res / geneCount**0.5
    return res


def countGene2(a):
    ''' count number of genes in a genome,
    from mata data file'''
    fn = metaFiles[a]
    with open(fn) as f:
        data = json.loads(f.read())
    res = sum( chrom['gene_count'] for chrom in data['chromosomes'] )
    return res


def countChromosome2(a):
    fn = metaFiles[a]
    with open(fn) as f:
        data = json.loads(f.read())
    res = len(data['chromosomes'])
    return res
    
    
def kernel3(ids):
    for i in ids:
        for j in ids:
            k = kernel3ij(i,j)
            # then combine all chrom matrices
            return k,[]


def kernel3ij(a,b):
    
    if a==b:
        count = countGene2(a)
        return np.eye(count)
    else:
        count1 = countChromosome2(a)
        count2 = countChromosome2(b)
        res = np.zeros([count1, count2])
        
        if a+'_'+b in ksFiles:
            fn = ksFiles[a+'_'+b]
        elif b+'_'+a in ksFiles:
            fn = ksFiles[b+'_'+a]
            
        with open(fn) as f:
            for line in f:
                if not (line.startswith('#')
                        or line.startswith('NA')
                        or line.startswith('undef')):
                    s = line.split('\t')
                    ks, chr1, chr2 = float(s[0]),s[2],s[6]
                    print ks, chr1, chr2
                    
                    
                    
    

if __name__=='__main__':
    print kernel2(25571, 11691)
