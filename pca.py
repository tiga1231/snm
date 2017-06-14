import numpy as np
from config import *
from kernel import genomeKernelMatrix, chromosomeKernelMatrix, chromosomeKernel

from sklearn.decomposition import KernelPCA,PCA


np.set_printoptions(precision=2)


def pca(level, ids, lam=2):
    if level==2:
        return pca2(ids, lam)
    if level==3:
        return pca3(ids)

        
def pca2(ids, lam=2):
    x = genomeKernelMatrix(ids, lam)
    print x
    pca = KernelPCA(n_components=2,kernel='precomputed')
    #pca = PCA(n_components=n)
    xCap = pca.fit_transform(x)

    '''
    from scipy.spatial.distance import pdist, squareform
    d2 = squareform(pdist(xCap))
    print d2
    import matplotlib.pyplot as plt
    plt.subplot(131)
    plt.imshow(-d2)
    plt.subplot(132)
    plt.imshow(-d2**2)
    plt.subplot(133)
    plt.imshow(x)'''
    tags = [genomeTags[i] for i in ids]
    
    #print x
    #print t
    #print pca.explained_variance_ratio_
    return xCap, tags, x
    
    
def pca3(ids):
    #test
    k, tags = chromosomeKernel(ids[0], ids[1])
    #real
    #k, tags = chromosomeKernelMatrix(ids)
    pca = KernelPCA(n_components=2,kernel='precomputed')
    xCap = pca.fit_transform(k)
    return xCap, tags
            
