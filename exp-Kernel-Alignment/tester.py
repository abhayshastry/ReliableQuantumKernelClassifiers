import pandas as pd
from  tqdm  import tqdm
from sklearn.svm  import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
import sys
from pathlib import Path
# As PosixPath, probably won't work on Windows
sys.path.append(Path(__file__).parent)
from q_kernels import *
from exp_utils import *
from qml_utils import *
from math_utils import *
#from data_gen import *

n = 5
qc, params = parametric_quantum_kernel_1(n)
#qc.draw(output = "latex_source")
#qc.draw(output = "latex", filename = "circuit.pdf")

np.random.seed(0)
C =  1
N_tot = 200
m = 15
m_sv = 50
beta =  2*C*(np.random.rand(m_sv) -0.5)
b = -0.1
margin = 0.0
X,y, X_sv, y_sv = data_gen(beta, b, margin , qc, params, n ,N_tot, scale =100, balance = 0.0)

X_test = X[m:N_tot]
y_test = y[m:N_tot]
X = X[:m]
y = y[:m]
quantum_kernel = havlicek_kernel #Havlicek Kernel
if quantum_kernel == havlicek_kernel:
    params = None

node = device_wrapper(n, quantum_kernel) 
K= quantum_kernel_matrix(X, node, weights = params)


np.random.seed(0)


K_train, y_train = return_kernel(("QAOA", "Generated"), m , bias = 0.3)





"""

W =  [q_features_from_circ(x, qc, params) for x in X]
W = np.hstack(W)
K =  np.dot(W.T,  np.conj(W))
K = np.real(np.einsum("ij, ij  -> ij", K,np.conj(K)))
N_meas_list = [10, 100, 200]
df = N_vs_emp_risk(K, y, N_meas_list , K, y, N_trials = 15)
high_prob_remp = []
delta = 0.2

for N in N_meas_list:
    high_prob_remp.append( high_prob_upper( df.loc[N,:]["Emp_risk"], delta ) )


clf =  SVC(kernel='precomputed', C = C)
clf.fit(K, y)
R_star = 1 - np.mean(y*clf.decision_function(K) > 0.5)

N_star_val =  N_star(K, y,  K, y, R_star,
                 N_list =  np.linspace(5,500, 10 , dtype = int) ,  N_trials = 10, delta = 0.2 , C = C,  Training = "Classical"  )

print(f"N_star:{N_star_val}")
"""
