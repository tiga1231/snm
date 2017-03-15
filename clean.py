from glob import glob
import os
import shutil

def getGeneCount():
    savePath = 'ks/geneCount.txt'
    with open(savePath,'w') as f:
        f.write('id,geneCount,name\n')
    for filename in glob('ks/fasta/*.*'):
        with open(filename) as f: 
            f.seek(-5555,2)
            count = f.readlines()[-2].split('||')[-1].rstrip()
            gid = filename.split('/')[-1].split('-')[0]
            print filename
            print gid
            print 'gene count:', count
        with open('ks/geneCount.txt','a') as f:
            f.write(gid+','+count+',\n')
#getGeneCount()

def cleanChr(g1 = '11691', g2 = '25571'):
##    remove tmp files
    shutil.rmtree('ks/cleanChrTmp/')
    os.mkdir('ks/cleanChrTmp/')
    g1 = str(g1)
    g2 = str(g2)
    for fpath in glob('ks/real/*.*'):
        fn = fpath.split('/')[-1]
        if g1 in fn and g2 in fn:
            with open(fpath) as f:
                lines = [i for i in f.readlines() \
                     if not i.startswith('#') \
                     and not i.startswith('NA')\
                     and not i.startswith('undef')]
                for line in lines:
                    a = line.split('\t')
                    ks,chr1,chr2 = a[0],a[2][1:],a[6][1:]
                    with open('ks/cleanChrTmp/'
                              +chr1+'|'+chr2,'a') as fout:
                        fout.write(line)
                        #fout.write('1e99' + '\t' + chr1 + '\t' + chr2 + '\n')
            break
##  remove tmp files
##    shutil.rmtree('ks/cleanChrTmp/')
##    os.mkdir('ks/cleanChrTmp/')
##cleanChr()

def cleanKS(gids = ['11691', '7057', '28918', '25571', '4242']):
    '''
        gids - genomes (IDs) of interest
    '''
    for f in glob('ks/cleaned/*.*'):
        os.remove(f)
    
    for filename in glob('data/real/*.ks'):
    ##    get genome IDs from filename
        g1, g2 = filename.split('/')[-1].split('.')[0].split('_')
    ##    if this file is of interest
        if g1 in gids and g2 in gids:
            
            saveName = 'data/cleaned/' + g1 + '_' + g2 + '.ks'
            with open(filename) as f:
                l = [i for i in f.readlines() \
                             if not i.startswith('#') \
                                 and not i.startswith('NA')\
                                 and not i.startswith('undef')\
                     ]

            l = [\
            i.split('\t')[0]+'\t' \
            +'-'.join(i.split('\t')[4:6])+'\t' \
            +'-'.join(i.split('\t')[8:10]) \
            for i in l]


            with open(saveName,'w') as f:
                f.write('\n'.join(l))


##cleanKS()
