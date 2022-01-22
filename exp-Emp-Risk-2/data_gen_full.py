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
from q_kernels import *
from itertools import product
from exp_utils import *
from qml_utils import *
from math_utils import *


def stochastic_root_finding(f, x_range = (1, 4000), N_queries = 40, N_trials = 100, delta = 0.1, domain_f = "Int"):
    """
    Find smallest x such that Pr(f(x) <  0) > 1 - delta
    """
    def  high_p_fn(x):
        List = [f(x) for _ in range(N_trials)]
        tol =  1.44*np.sqrt( np.var( List)) ##Don't ask :-/
        f_val = high_prob_upper(List, delta)
        return 0.0 if  abs(f_val) < tol else f_val

    def next_val(x1, x2, f1, f2):
        if domain_f == "Int":
            x_h = int(0.5*(x1 + x2))
        if domain_f == "Real":
            x_h = 0.5*(x1 + x2)
        f_h = high_p_fn(x_h)
#        if abs(f_h) < tol:
#           return "Terminated", x_h, f_h
        if f1*f_h <= 0:
            return x1, x_h, f1, f_h
        else:
            return x_h, x2, f_h, f2

    xi = x_range[0]
    fi = high_p_fn(xi)
    xe = x_range[1]
    fe = high_p_fn(xe)

    for t in range(N_queries):
        A = next_val(xi, xe, fi, fe)
        print(f"xvals:{A[:2]}, f_vals: {A[2:]}")
        if A[1] - A[0] < 2:
            break

        else:
            xi, xe, fi, fe = A

    xr = A[0] if A[2] < A[3] else A[1]
    fr = min(A[2], A[3])
    return xr, fr

def find_N_star(args, N_trials = 100,  delta = 0.1):
    key, m, C = args
    print(f'[JOB {args}]', flush=True)
    K_train, y_train = return_kernel(key, m, bal_tol = 0.02 )
    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = np.linspace(1/m, 200, 50))

    clf =  SVC(kernel='precomputed', C = C)
    clf.fit(K_train, y_train)
    f_pred_exact =  clf.decision_function(K_train).flatten()
    margin_err = 1 - np.mean(y_train*f_pred_exact > 1.0  )
    print(f"margin_err = {margin_err}")

    def fn1(N):
        K_N =  kernel_estimate(K_train, N)
        y_pred_N = np.sign( clf.decision_function(K_N).flatten())
        return   0.5*np.mean(np.abs(y_pred_N - y_train.flatten())) - margin_err


    A =  stochastic_root_finding(fn1,  N_queries = 100,  N_trials = N_trials, delta= delta)
    return A


def run(args):
    key, m, C = args
    print(f'[JOB {args}]', flush=True)
    K_train, y_train = return_kernel(key, m, bal_tol = 0.02 )

    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = np.linspace(0.1, 10, 11))

    clf =  SVC(kernel='precomputed', C = C)
    clf.fit(K_train, y_train)
    m_sv =sum(clf.n_support_)
    f_pred_exact =  clf.decision_function(K_train)
    A = find_N_star(args)

    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            data = {}

    data[args] = {
        'N_star_output':A , 'f_pred_exact':f_pred_exact,  'y_train':y_train, 'C_val': C, 'm_sv': m_sv, 'K_train':K_train}
   # print({'N_star':N_star_val, 'emp_risk': R_star})

    with open(path, 'w+b') as f:
        pickle.dump(data, f)

    print(f'[Finished {args}]', flush=True)

    return


kd_list = [("QAOA", "Checkerboard" ), ("Havliscek,2", "Two_Moons" ), ("Circ-Hubr,2", "Two_Moons" ),
 ("Circ-Hubr", "Generated" ), ("Havliscek,2", "Checkerboard" ),  ("QAOA,2", "Two_Moons" ),  ("Angle,2", "Two_Moons" ), ("Angle", "Generated" ),  ("QAOA" ,"Generated" ), ("Havliscek", "Generated" ),
 ("QAOA,2", "SymDonuts" ), ("QAOA", "Two_Moons" ),  ("Circ-Hubr,2", "Checkerboard" ), ("QAOA,2", "Checkerboard" )]


k_list = ["Havliscek", "QAOA", "Angle", "QAOA,2", "Havliscek,2", "Angle,2"]
#d_list = ["Two_Moons" , "Checkerboard", "SymDonuts"]
d_list = ["Gen,2", "Gen,5"]

indices = [
    ('kernel, dataset', [a for a in product(k_list[:3], d_list)]  ),
    ('m', [60, 120, 300]),
    ("C", ["optimal", 0.1, 1 ] ) ##Add optimal C as well
]

#column_names = ['N_list', 'y_pred_list', 'y_train']

path = "data_full_new.pkl"


t = time.time()
rerun = False

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






