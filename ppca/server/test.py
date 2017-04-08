from server import c1,c2,c3, jac1, jac2, jac3
import numpy as np

beta = np.array([[1,2],[3,4],[5,6]])

print c1(beta)
print c2(beta)
print c3(beta)

beta = beta.flatten()

print jac1(beta)
print jac2(beta)
print jac3(beta)

