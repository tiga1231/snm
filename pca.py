import numpy as np
from config import *
from kernel import genomeKernelMatrix, chromosomeKernelMatrix, chromosomeKernel

from sklearn.decomposition import KernelPCA,PCA


np.set_printoptions(precision=2)


def pca(level, ids):
    if level==2:
        return pca2(ids)
    if level==3:
        return pca3(ids)

        
def pca2(ids):
    x = genomeKernelMatrix(ids)
    print x
    pca = KernelPCA(n_components=2,kernel='precomputed')
    #pca = PCA(n_components=n)
    xCap = pca.fit_transform(x)
    tags = [genomeTags[i] for i in ids]
    #print x
    #print t
    #print pca.explained_variance_ratio_
    return xCap, tags
    
    
def pca3(ids):
    #test
    k, tags = chromosomeKernel(ids[0], ids[1])
    #real
    #k, tags = chromosomeKernelMatrix(ids)
    pca = KernelPCA(n_components=2,kernel='precomputed')
    xCap = pca.fit_transform(k)
    return xCap, tags
            
