import numpy as np
import pandas as pd
from  tqdm  import tqdm
from sklearn.svm   import SVC
import sys
#from pathlib import Path
# As PosixPath, probably won't work on Windows
#sys.path.append(Path(__file__).parent)
import time, os, itertools, pickle, sys
srcpath = os.path.abspath(os.path.join(os.path.abspath(''),  '..',  'src'))
sys.path.append(srcpath)
from itertools import product
from q_kernels import *
from exp_utils import *
from qml_utils import *
from math_utils import *

N_trials = 100
eta = 1

rerun = True




def run(args):
    key, m, m_test, C, N, mode = args
    print(f'[JOB {args}]', flush=True)
    K, y = return_kernel(key, m + m_test )
    K_train = K[:m, :m]
    y_train = y[:m]
    K_test_train =  K[m:, :m]
    y_test = y[m:]

    if C  == "1/sqrt(m)":
        C = 1/np.sqrt(m)
    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = [0.1,1,2,5])

    clf =  SVC(kernel='precomputed', C = C)
    clf.fit(K_train, y_train)
    m_sv =sum(clf.n_support_)
    y_train_pred = np.sign(clf.decision_function(K_train))
    y_test_pred = np.sign(clf.decision_function(K_test_train))
    Y_train_pred_tensor = np.zeros((N_trials, m))
    Y_test_pred_tensor = np.zeros((N_trials, m_test))

    for ind in tqdm(range(N_trials)):
        ##Randomness in only testing
        if mode == "test_only":
            Y_train_pred_tensor[ind, :] =  np.sign(clf.decision_function(kernel_estimate(K_train, N)))
            Y_test_pred_tensor[ind, :] =  np.sign(clf.decision_function(kernel_estimate(K_test_train, N)))
        if mode == "test_train":
            clf =  SVC(kernel='precomputed', C = C)
            clf.fit(kernel_estimate(K_train, N), y_train)
            Y_train_pred_tensor[ind, :] =  np.sign(clf.decision_function(kernel_estimate(K_train, N)))
            Y_test_pred_tensor[ind, :] =  np.sign(clf.decision_function(kernel_estimate(K_test_train, N)))






    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            data = {}

    data[args] = {
        'y_train':y_train, 'C_val': C, 'm_sv': m_sv, 'K_train':K_train,
        'y_test':y_test, 'K_test_train':K_test_train,
        'y_train_pred':y_train_pred,
        'y_test_pred':y_test_pred,
        'y_train_pred_random':Y_train_pred_tensor,
        'y_test_pred_random':Y_test_pred_tensor
    }
   # print({'N_star':N_star_val, 'emp_risk': R_star})

    with open(path, 'w+b') as f:
        pickle.dump(data, f)

    print(f'[Finished {args}]', flush=True)

    return


#kd_list = [("QAOA", "Checkerboard" ), ("Havliscek,2", "Two_Moons" ), ("Circ-Hubr,2", "Two_Moons" ),
# ("Circ-Hubr", "Generated" ), ("Havliscek,2", "Checkerboard" ),  ("QAOA,2", "Two_Moons" ),  ("Angle,2", "Two_Moons" ), ("Angle", "Generated" ),  ("QAOA" ,"Generated" ), ("Havliscek", "Generated" ),
#           ("QAOA,2", "SymDonuts" ), ("QAOA", "Two_Moons" ),  ("Circ-Hubr,2", "Checkerboard" ), ("QAOA,2", "Checkerboard" )]

#kd_list =  



indices = [
    ('kernel, dataset', kd_list),
    ('m', [60]),
    ('m_test', [60]),
    ("C", [ 1 ] ), ##Add optimal C as well
    ("N", np.ceil(np.logspace(1,4, num=10)).astype(np.int32)),
    ("mode",["test_train", "test_only"]) 
]

#column_names = ['N_list', 'y_pred_list', 'y_train']

path = "data_full_new.pkl"


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
if rerun:
    todos = args
else:
    todos = list(set(args) - data.keys())

print(f'{len(todos)} of {len(args)} data entries are missing. Starting to run ...')

for job in todos:
    run(job)
    print(f"Size of results {len(data)}")

print(f'All jobs finished in {time.time()-t:.3f}s')






