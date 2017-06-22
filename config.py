from glob import glob
import os

dataRoot = 'data/'
ksRoot = dataRoot + 'ks/'
metaRoot = dataRoot + 'meta/'
fastaRoot = dataRoot + 'fasta/'

genomeTagFile = dataRoot + 'myTags.txt'
with open(genomeTagFile) as f:
    genomeTags = dict(line.replace('\n','').split(',') 
                        for line in f.readlines())
genomeTagFile = dataRoot + 'plasmodiumTags.txt'
with open(genomeTagFile) as f:
    genomeTags.update( dict(line.replace('\n','').split(',')[::-1] 
                        for line in f.readlines()) )

ksFiles = glob(ksRoot+'*.ks')
ksFiles = dict([
    (os.path.split(f)[1].split('.')[0], f)        
    for f in ksFiles])

metaFiles = glob(metaRoot+'*')
metaFiles = dict([
    (os.path.split(f)[1], f)
    for f in metaFiles
    ])

if __name__ == '__main__':
    #print ksFiles
    #print metaFiles
    print genomeTags

