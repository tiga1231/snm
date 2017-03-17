from glob import glob

dataRoot = 'data/'
ksRoot = dataRoot + 'ks/'
metaRoot = dataRoot + 'meta/'
fastaRoot = dataRoot + 'fasta/'

genomeTags = dataRoot + 'myTags.txt'
with open(genomeTags) as f:
    genomeTags = dict(line.replace('\n','').split(',') for line in f.readlines())

ksFiles = glob(ksRoot+'*')
ksFiles = dict([
    (f.split('/')[-1].split('.')[0], f)        
    for f in ksFiles])

metaFiles = glob(metaRoot+'*')
metaFiles = dict([
    (f.split('/')[-1], f)
    for f in metaFiles
    ])

