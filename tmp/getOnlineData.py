import urllib2

url = 'https://genomevolution.org/coge/SynMap.pl?dsgid1=28041;dsgid2=11691;D=20;g=10;A=5;w=0;b=1;ft1=1;ft2=1;dt=geneorder;ks=1;autogo=1'

res = urllib2.urlopen(url)
with open('res.html','w') as f:

    f.write(res.read())
