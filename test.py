import numpy as np
from kernel import centerize as c

a = np.array([[2,1,3],[4,3,2],[5,4,3]])
print a
print c(a)
