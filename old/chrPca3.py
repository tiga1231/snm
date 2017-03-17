from glob import glob
import json
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

from kernel import kernel2
from clean import cleanChr
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

gids = ['11691', '25571']#,'7057','28918','4242','28814']
meta = {}
chrNames = [0,0]
for j,gid in enumerate(gids):
    with open('data/meta/'+gid) as meta_file:    
        meta[gid] = json.load(meta_file)
        geneCountSorted = sorted(meta[gid]['chromosomes'],
                key = lambda x:x['gene_count'],
                reverse = True)
        chrNames[j] = [i['name'] for i in geneCountSorted[:25]]
        
                
#cleanChr(gids[0],gids[1])

chrNamesComb = chrNames[0] + chrNames[1]
K = pd.DataFrame(
        data = np.eye(len(chrNamesComb)), 
        index = chrNamesComb, 
        columns = chrNamesComb)

#build K, the chr ks matrix, each entry is a dot product
#measure of a chromosome pair
for fpath in glob('data/cleanChrTmp/*.*'):
    s = fpath.split('/')[-1].split('|')
    gid1 = s[0].split('_')[0]
    gid2 = s[1].split('_')[0]
    #print gid1, gid2
    chr1 = s[0].split('_')[-1]
    chr2 = s[1].split('_')[1:]
    chr2 = '_'.join(chr2)
    #print chr1, chr2
    if chr1 in chrNames[0] and chr2 in chrNames[1]:
        df = pd.read_csv(fpath,
                        header = None,
                        sep = '\t',
                        )
    
        n1 = filter(lambda o: o['name']==chr1, meta[gid1]['chromosomes'])
        n2 = filter(lambda o: o['name']==chr2, meta[gid2]['chromosomes'])
        #print n1,n2
        n1 = n1[0]['gene_count'] if len(n1)>0 else (  len(set(df[1])) or 1  )
        n2 = n2[0]['gene_count'] if len(n2)>0 else (  len(set(df[2])) or 1  )
        #print n1,n2
        k = kernel2(df, 1e-99, n1, n2)
        K[chr1][chr2] = k
        K[chr2][chr1] = k


#np.savetxt('k.txt',K.as_matrix(), '%2.2f', ', ','\n')
pca = PCA(n_components = 2)
pca.fit(K)
x = pca.transform(K)

categories = [0 if c in chrNames[0] 
        else 1 for c in K.index] 

x = [{'x':d[0], 'y':d[1], 
        'cat':categories[i],
        'tag':chrNamesComb[i]} for i,d in enumerate(x)]


with open('3.js', 'w') as f:
    f.write('var data = \n')
    json.dump(x,f, indent = 2)
