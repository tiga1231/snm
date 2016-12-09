import numpy as np
import pandas as pd

##kernel of a single pair of points
def k(ks, sigma=1):
    '''ks - distance measure,
        sigma - sensitivity to distance
    '''
##    subject to change, depending on properties of ks
##    smaller sigma, sharper
    return np.exp(- ks * sigma)


def buildMatrix(df):
    '''convert from ks list to ks matrix'''
    KS = pd.DataFrame(data = 1e100,
                  index = set(df[:][1]),
                  columns = set(df[:][2]),
                  )
    ####NOTE: EXTREMELY SLOW
    for i in df.iterrows():
        row = i[1]
        KS[row[2]][row[1]] = row[0]
    return np.array(KS, dtype=np.float64)


#kernel of two sets of points
def kernel(x, sigma=1):
    if type(x) is str:
        return kernel3(x, sigma)
    elif type(x) is pd.core.frame.DataFrame:
        return kernel2(x, sigma)
    elif type(x) is np.ndarray:
        return kernel1(x, sigma)


def kernel1(KS, sigma=1):
    '''KS - distance matrix between two sets'''
##    weight matrix
    W = np.ones(KS.shape) / (KS.shape[0] * KS.shape[1])
    return np.sum(W * k(KS, sigma))


def kernel2(df, sigma=1, n1 = None, n2 = None):
    if n1 is None:
        n1 = len(set(df[1])) or 1
    if n2 is None:
        n2 = len(set(df[2])) or 1
        
    return np.sum(k(df[0], sigma)/(n1*n2)**0.5)


def kernel3(ksfile, sigma=1):
    try:
        df = pd.read_csv(ksfile,
                        header = None,
                        sep = '\t',
                        )
    except pd.io.common.EmptyDataError:
        return 0
    
    g1, g2 = ksfile.split('/')[-1].split('.')[0].split('_')
    g1, g2 = int(g1), int(g2)
    print g1,g2
    genomeInfo = pd.read_csv('ks/geneCount.txt',
                             index_col = 'id')
    g1Count = genomeInfo['geneCount'][g1]
    g2Count = genomeInfo['geneCount'][g2]
    print g1Count,g2Count 
    return kernel2(df, sigma, g1Count, g2Count)

########################################################
#### read ks.txt

if __name__ == '__main__':
    print kernel3('ks/cleaned/11691_7057.ks')
##    print kernel('ks/ksBB.txt')
##    
##    
##    
##    df = pd.read_csv('ks/ksBB.txt',
##                    header = None,
##                    sep = '\t',
##                    )
##    print kernel(df)
##
##    KS = buildMatrix(df)
##    print kernel(KS)
    
    













