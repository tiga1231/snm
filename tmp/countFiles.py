from glob import glob

spec = []
for filepath in glob('../ks/real/*'):
    name = filepath.split('/')[-1]
    ids = name.split('.')[0].split('_')
    spec += ids

print len(spec)
for i in set(spec):
    print i, spec.count(i)
