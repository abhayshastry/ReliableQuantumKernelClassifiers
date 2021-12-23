import numpy as np
import pandas as pd
from  tqdm  import tqdm
from sklearn.svm   import SVC
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






def run(args):
    key, m, C = args
    print(f'[JOB {args}]', flush=True)
    K_train, y_train = return_kernel(key, m )
    if C  == "1/sqrt(m)":
        C = 1/np.sqrt(m)
    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = np.linspace(1/m, 200, 50))

    clf =  SVC(kernel='precomputed', C = C)
    clf.fit(K_train, y_train)
    m_sv =sum(clf.n_support_)
    eta = 10000
    N_ker = N_from_eta(K_train, eta, 0.1)
    print(f"N_ker={N_ker}")

    N_list = np.linspace(2,3000, 15 , dtype = int)
    df_N = N_vs_y_pred (K_train, y_train,
                        N_list, K_train, y_train, C =C, N_trials = 20 )
    f_pred_exact =  clf.decision_function(K_train)
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            data = {}

    data[args] = {
        'N_vs_y_pred':df_N, 'f_pred_exact':f_pred_exact,  'y_train':y_train, 'C_val': C, 'm_sv': m_sv, 'N_ker': N_ker}
   # print({'N_star':N_star_val, 'emp_risk': R_star})

    with open(path, 'w+b') as f:
        pickle.dump(data, f)

    print(f'[Finished {args}]', flush=True)

    return


k_list = [ "Havliscek", "Circ-Hubr" ,"QAOA", "Angle", "Circ-Hubr,2" , "Havliscek,2", "QAOA,2", "Angle,2"]
d_list = ["Two_Moons" , "Checkerboard","SymDonuts"]

#k_list = ["Circ-Hubr,3" , "Havliscek,3", "QAOA,3", "Angle,3"]
#d_list = ["Checkerboard"]


indices = [
    ('kernel, dataset', [a for a in product(k_list[:4],["Generated"])] + [a for a in product(k_list, d_list)]),
   # ('kernel, dataset',  [a for a in product(k_list, d_list)]),
    ('m', [120]),
    ("C", ["1/sqrt(m)", "optimal" ]) ##Add optimal C as well
]

#column_names = ['N_list', 'y_pred_list', 'y_train']

path = "data_full_new.pkl"


t = time.time()
rerun = True

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
if rerun:
    todos = args 
else:
    todos = list(set(args) - data.keys())

print(f'{len(todos)} of {len(args)} data entries are missing. Starting to run ...')

for job in todos:
    run(job)
    print(f"Size of results {len(data)}")

print(f'All jobs finished in {time.time()-t:.3f}s')







