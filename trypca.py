import numpy as np
from numpy import linalg as la
from scipy.spatial.distance import pdist, squareform

##from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

x0 = np.array([1,2,1,1,1,0,0,0,2,3,5,6,4,5,3,2,1,2,3,8]).reshape([-1,2])
print x0


distSq = squareform(pdist(x0))**2
print distSq
n = distSq.shape[0]
h = np.eye(n)-np.ones([n,n])/n

xtx = - h.dot(distSq).dot(h)/2.0
w,v = la.eig(xtx)

print xtx
print w
print v

x = xtx.dot(v)

plt.subplot(121)
plt.scatter(x0[:,0], x0[:,1])
plt.subplot(122)
plt.scatter(x[:,0], x[:,1])
plt.show()
