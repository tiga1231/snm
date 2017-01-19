import numpy as np
import matplotlib.pyplot as plt
from glob import glob
x = []

filename = glob('../ks/real/*')[7]
with open(filename) as f:
    for line in f:
        if line.startswith('#'):
            continue
        ks = line.split('\t')[-2]
        ks = float(ks)
        if ks > 0.0001:
            x.append(ks)

x = np.array(x)
plt.hist(x, bins=20)
plt.title(filename.split('/')[-1])

plt.show()
