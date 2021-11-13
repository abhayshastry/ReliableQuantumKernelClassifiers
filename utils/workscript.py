import math_utils as mu
import numpy as np
import qml_utils as qu
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y = True)
X = X[:10]
y = y[:10]
node = qu.device_wrapper(4, qu.havlicek_kernel)
np.set_printoptions(precision=2)
kernel = node(X[0],X[1],4)
print(kernel)
kernel_matrix = qu.quantum_kernel_matrix(X, node)
print(kernel_matrix)
skm = mu.skm(kernel_matrix, shots =1, N_trials = 100)
b = np.zeros(kernel_matrix.shape)
for a in skm:
    b=b+a
b=b/len(skm)
#print(skm)
print(b)
print(kernel_matrix)
