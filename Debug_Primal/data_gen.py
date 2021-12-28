import numpy as np
import pandas as pd
from  tqdm  import tqdm
from sklearn.svm   import SVC
from sklearn.datasets import make_moons
import random
import sys
#from pathlib import Path
# As PosixPath, probably won't work on Windows
#sys.path.append(Path(__file__).parent)
import time, os, itertools, pickle
from itertools import product
from q_kernels import *
from exp_utils import *
from qml_utils import *
#from math_utils import *




## Test 1. So see what combinations of paramters cause the optimizer to fail
rerun = True
path = "data.pkl"



def run_rob(args):
    print(f'[JOB {args}]', flush=True)

    key, m, delta_1, delta_2, C, shots, circuit_type, training_type, trials = args
    if not delta_1:
        delta_1 = 1
    if not delta_2:
        delta_2 = m

    K_train, y_train = return_kernel(key, m )
    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = [0.5,1,3,5,10])

    try:
        ### Here we try to run the robust program with two differnt values of numerical shift
        try:
            beta, b, SV = primal_rob_no_dummy(K_train, y_train, delta_1 = delta_1, C = C,
                                              delta_2 = delta_2/m, shots = shots,  circuit_type = circuit_type)
        except:
            beta, b, SV = primal_rob_no_dummy(K_train, y_train, delta_1 = delta_1, C = C,
                                              delta_2 = delta_2/m, shots = shots,  circuit_type = circuit_type,
                            numerical_shift = 1e-3 )
        works = True
        print(f" Optimization works: {works}")
    except:
        beta = None
        b = None
        works = False
        print(f" Optimization works: {works}")

    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            data = {}

    data[args] = {
        'works':works, 'beta':beta, 'b':b}

    with open(path, 'w+b') as f:
        pickle.dump(data, f)

    print(f'[Finished {args}]', flush=True)

    return

k_list = [ "Havliscek", "Circ-Hubr" ,"QAOA", "Angle", "Circ-Hubr,2" , "Havliscek,2", "QAOA,2", "Angle,2"]
d_list = ["Two_Moons" , "Checkerboard","SymDonuts"]

#k_list = ["Circ-Hubr,3" , "Havliscek,3", "QAOA,3", "Angle,3"]
#d_list = ["Checkerboard"]


indices = [
  #  ('kernel, dataset', [a for a in product(k_list[:4],["Generated"])] + [a for a in product(k_list, d_list)]),
    ('kernel dataset',  [("Havliscek","Generated"), ("Havliscek,2","Checkerboard")]),
    ('m', [60,120]),
    ("delta_1", [ 0.01, 0.1 ]),
    ("delta_2", [ 0.01, 0.1]),
    ("C", ["optimal"]),
    ("shots", [20,100, 300, 600, 1200, 1800, 2500, 3000]),
    ("method", ["gates"]),
    ("train", ["exact"]),
    ("nTrials", [10])
]







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
random.shuffle(todos)
for job in todos:
    run_rob(job)
    print(f"Size of results {len(data)}")

print(f'All jobs finished in {time.time()-t:.3f}s')



## Test 2: To collect data about the differnce betweent the coefficents returned by the our code and svc
rerun = True
path = "data_svc_rob_comparison.pkl"



def run_rob(args):
    print(f'[JOB {args}]', flush=True)

    key, m, delta_1, delta_2, C, shots, circuit_type, training_type, trials = args
    if not delta_1:
        delta_1 = 1.0
    if not delta_2:
        delta_2 = m

    K_train, y_train = return_kernel(key, m )
    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = [0.5,1,3,5,10])

    beta_vals= np.zeros(m)
    nominal_classifier = SVC(kernel="precomputed",C=C).fit(K_train,y_train)
    beta_svc = nominal_classifier.dual_coef_.flatten()
    b_svc = nominal_classifier.intercept_[0]
    SV = nominal_classifier.support_
    beta_vals[SV] = beta_svc

    try:
        try:
            beta, b, SV = primal_rob_no_dummy(K_train, y_train, delta_1 = delta_1, C = C,
                                              delta_2 = delta_2/m, shots = shots,  circuit_type = circuit_type)
        except:
            beta, b, SV = primal_rob_no_dummy(K_train, y_train, delta_1 = delta_1, C = C,
                                              delta_2 = delta_2/m, shots = shots,  circuit_type = circuit_type,
                            numerical_shift = 1e-3 )
        works = True
        beta_err = np.linalg.norm(beta - beta_vals, ord = np.inf)/ np.linalg.norm(beta_vals, ord = np.inf)
        b_err =  np.abs(b - b_svc)/np.abs(b_svc)
        print(f" Optimization works: {works}")
    except:
        works = False
        beta_err = None
        b_err = None
        beta = None
        print(f" Optimization works: {works}")

    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError:
            data = {}

    data[args] = {
        'works':works, 'beta_err_rel': beta_err,  'b_err_rel': b_err,
                    "C_val": C, "beta_rob":beta, "beta_vals":beta_vals}

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
 #   ('kernel dataset',  [("Havliscek","Generated"), ("Havliscek,2","Checkerboard")]),
    ('m', [60,120]),
    ("delta_1", [ False ]),
    ("delta_2", [  False ]),
    ("C", ["optimal"  ]),
    ("shots", [20]),
    ("method", ["gates"]),
    ("train", ["exact"]),
    ("nTrials", [10])
]







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
random.shuffle(todos)
for job in tqdm(todos):
    run_rob(job)
    print(f"Size of results {len(data)}")

print(f'All jobs finished in {time.time()-t:.3f}s')

