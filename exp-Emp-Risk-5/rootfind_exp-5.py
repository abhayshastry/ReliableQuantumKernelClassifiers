# script actually implies a high probability upper bound as given by the theorem in the paper



import numpy as np
import pandas as pd
from  tqdm  import tqdm
from sklearn.svm   import SVC
import sys
import time, os, itertools, pickle, sys
srcpath = os.path.abspath(os.path.join(os.path.abspath(''),  '..',  'src'))
sys.path.append(srcpath)
from q_kernels import *
from itertools import product
from exp_utils import *
from qml_utils import *
from math_utils import *
path = "data_full_new.pkl"

with open(path, 'rb') as f:
    data = pickle.load(f)

N_trials = 100
delta = 0.1

def emp_prob(K_train, y_train, margin_err, N, C):
    if N ==0:
        N = 1
    clf =  SVC(kernel='precomputed', C = C)
    List = []
    clf.fit(K_train, y_train)
    np.random.seed(0)
    for _ in range(N_trials):
        K_N =  kernel_estimate(K_train, N)
        y_pred_N = np.sign( clf.decision_function(K_N).flatten())
        List.append(0.5*np.mean(np.abs(y_pred_N - y_train.flatten())))
    return np.mean( List <= margin_err )

key_m_C_list = [ (("QAOA","Two_Moons"), 120, 1 ), (("QAOA","Two_Moons"), 120, 5 ),
                 (("Havliscek","Gen,2"),60,5),  (("Havliscek","Gen,2"),60,1),
                 (("Angle,2","Circles"),120,5),  (("QAOA,2","Circles"),120,5),
                 (("Angle,2","Circles"),120,1),  (("QAOA,2","Circles"),120,1), ]


for k in tqdm(data):

    key, m ,C = k
  #  if k not in key_m_C_list:
  #      continue
    V = data[k]
    f_pred = V['f_pred_exact']
    y_train = V['y_train']
    K_train = V['K_train']
    N_star = V["N_star_output"][0]
    margin_err = 1.0 - np.mean( f_pred*y_train > 1.0 - 1e-3)
    if margin_err > 0.2 or N_star is None:
        continue
    print( f"{k}")
    print(f"N_star_margin = {N_star}")
    N = int(N_star)
    N_start = (N // 50)*50
    pl = emp_prob (K_train, y_train, margin_err, N_start,C)
    print(f"Current N: {N_start}, Current emp probability: { pl}")
    while pl < 1 -delta:
        N_start = N_start + 50
        pl = emp_prob (K_train, y_train, margin_err, N_start,C)
        print(f"Current N: {N_start}, Current emp probability: { pl}")

    data[k]["N_margin_recomputed"] = N_start
    print(f"True N_margin { N_start}")
    print("----------------------")


with open(path, 'w+b') as f:
    pickle.dump(data, f)

