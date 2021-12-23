import numpy as np
import pandas as pd
from  tqdm  import tqdm
from sklearn.svm   import SVC
from sklearn.datasets import make_moons
import sys
#from pathlib import Path
# As PosixPath, probably won't work on Windows
#sys.path.append(Path(__file__).parent)
import time, os, itertools, pickle
from itertools import product
from q_kernels import *
from exp_utils import *
from qml_utils import *
from math_utils import *

def return_kernel(key,m, n_qubits = 5, seed = 0):
    np.random.seed(seed)
    if key == ("Circ-Hubr", "Generated"):
        print(f"Generating data:{key}")
        qc, params = parametric_quantum_kernel_1(n_qubits)
        C = 1
        m_sv = 20
        beta =  2*C*(np.random.rand(m_sv) -0.5)
        b = -0.1
        margin = 0.0
        X,y, X_sv, y_sv = data_gen(beta, b, margin , qc, params, n_qubits ,m, scale =100, balance = 0.0)
        W =  [q_features_from_circ(x, qc, params) for x in X]
        W = np.hstack(W)
        K =  np.dot(W.T,  np.conj(W))
        K = np.real(np.einsum("ij, ij  -> ij", K,np.conj(K)))

    if key == ("Havliscek", "Generated"):
        print(f"Generating data:{key}")
        node = device_wrapper(n_qubits, havlicek_features)
        M = np.random.rand(2**n_qubits, 2**n_qubits) - 0.5
        M = M + M.T
        X,y = quantum_generate_dataset(m, n_qubits, M, node)
        node = device_wrapper(n_qubits, havlicek_kernel)
        K= quantum_kernel_matrix(X, node, weights = None)

    if key == ("Angle", "Generated"):
        print(f"Generating data:{key}")
        node = device_wrapper(n_qubits, angle_features)
        M = np.random.rand(2**n_qubits, 2**n_qubits) - 0.5
        M = M + M.T
        X,y = quantum_generate_dataset(m, n_qubits, M, node)
        node = device_wrapper(n_qubits, angle_kernel)
        K= quantum_kernel_matrix(X, node, weights = None)

    if key == ("QAOA", "Generated"):
        print(f"Generating data:{key}")
        node = device_wrapper(n_qubits, qaoa_features)
        M = np.random.rand(2**n_qubits, 2**n_qubits) - 0.5
        M = M + M.T
        L = 2
        weights = np.random.rand(L, 2*n_qubits)

        X,y = quantum_generate_dataset(m, n_qubits, M, node, weights = weights)
        node = device_wrapper(n_qubits, qaoa_kernel)
        K= quantum_kernel_matrix(X, node, weights = weights)


    if (key[1] == "Two_Moons") or (key[1] == "Checkerboard"):
        n_qubits = 2
        if key[1] == "Two_Moons":
            X, y =  make_moons(m)
            X = [X[i,:] for i in range(m)]
            y = 2*y - 1
        if key[1] == "Checkerboard":
            X,y = checkerboard(m)
            X = [X[i,:] for i in range(m)]

        if key[0] == "Angle":
            print(f"Generating data:{key}")
            node = device_wrapper(n_qubits, angle_kernel)
            K= quantum_kernel_matrix(X, node, weights = None)

        if key[0] == "Havliscek":
            print(f"Generating data:{key}")
            node = device_wrapper(n_qubits, havlicek_kernel)
            K= quantum_kernel_matrix(X, node, weights = None)

        if key[0] == "Circ-Hubr":
            print(f"Generating data:{key}")
            qc, params = parametric_quantum_kernel_1(n_qubits)
            W =  [q_features_from_circ(x, qc, params) for x in X]
            W = np.hstack(W)
            K =  np.dot(W.T,  np.conj(W))
            K = np.real(np.einsum("ij, ij  -> ij", K,np.conj(K)))
        if key[0] == "QAOA":
            L = 2
            weights = np.random.rand(L, 3)
            node = device_wrapper(n_qubits, qaoa_kernel)
            K= quantum_kernel_matrix(X, node, weights = weights)


    return K,y




def run(args):
    key, m, target, C = args
    print(f'[JOB {args}]', flush=True)
    K_train, y_train = return_kernel(key, m )
    print(f"K[10,1] = {K_train[10,1]}")
    if C  == "1/sqrt(m)":
        C = 1/np.sqrt(m)

    if C == "optimal":
        C = find_optimal_C(K_train,y_train)

    clf =  SVC(kernel='precomputed', C = C)
    clf.fit(K_train, y_train)

    if target[0] == "margin":
        R_star = 1 - np.mean(y_train*clf.decision_function(K_train) > target[1])
    if target[0] == "relative":
        R_star = 1 - (np.mean(y_train*clf.decision_function(K_train) > 0 )) * (1 - target[1])

    N_star_val =  N_star(K_train, y_train,  K_train, y_train, R_star,
                    N_list =  np.linspace(2,3000, 15 , dtype = int) ,  N_trials = 20, delta = 0.2 , C = C,  Training = "Classical"  )

    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            data = {}

    data[args] = {
        'N_star':N_star_val, 'emp_risk': R_star, 'C_val': C}
    print({'N_star':N_star_val, 'emp_risk': R_star})

    with open(path, 'w+b') as f:
        pickle.dump(data, f)

    print(f'[Finished {args}]', flush=True)

    return


k_list = ["Circ-Hubr" , "Havliscek", "QAOA", "Angle"]
d_list = ["Two_Moons" , "Checkerboard", "Generated"]


indices = [
    ('kernel, dataset', [("Circ-Hubr", "Generated")]
                         + [a for a in product(k_list, d_list)]),
    ('m', [60,  120]),
    ('target_err',[("margin", 0.5), ("relative", 0.05 )]),
    ("C", [5, 1, "1/sqrt(m)", "optimal" ]) ##Add optimal C as well
]

column_names = ['N_star']

path = "data.pkl"


t = time.time()

if not os.path.exists(path):
    with open(path, 'w+b') as f:
        pickle.dump({}, f)

level_names, level_values = zip(*indices)
args = list(itertools.product(*level_values))
with open(path, 'rb') as f:
    try:
        data = pickle.load(f)
    except EOFError:
        data = {}
todos = list(set(args) - data.keys())

print(f'{len(todos)} of {len(args)} data entries are missing. Starting to run ...')

for job in todos:
    run(job)
    print(f"Size of results {len(data)}")

print(f'All jobs finished in {time.time()-t:.3f}s')







