import time, os, itertools, pickle, sys
#srcpath = os.path.abspath(os.path.join(os.path.abspath(''),  '..',  'src'))
sys.path.append("../")
from q_kernels import *
from itertools import product
from exp_utils import *
from qml_utils import *
from math_utils import *


key =  ("Angle", "Gen,2")
def stochastic_root_finding(f, x_range = (1, 4000), N_queries = 40, N_trials = 100, delta = 0.1/2, domain_f = "Int"):
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

def find_N_star(args, N_trials = 100,  delta = 0.05, bal_tol = 0.01):
    key, m, C = args
    print(f'[JOB {args}]', flush=True)
    K_train, y_train = return_kernel(key, m, bal_tol = bal_tol)
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

    def  high_p_fn(x):
        List = [fn1(x) for _ in range(N_trials)]
        tol =  1.44*np.sqrt( np.var( List)) ##Don't ask :-/
        f_val = high_prob_upper(List, delta)
        return  List
        return 0.0 if  abs(f_val) < tol else f_val
    #return high_p_fn(8)

    A =  stochastic_root_finding(fn1,  N_queries = 100,  N_trials = N_trials, delta= delta)
    return A


m = 60
C = 1
args = (key, m ,C)
bal_tol = max( 2/m , 0.01)
K_train, y_train = return_kernel(key, m, bal_tol = bal_tol )
clf =  SVC(kernel='precomputed', C = C)
clf.fit(K_train, y_train)
m_sv =sum(clf.n_support_)
f_pred_exact =  clf.decision_function(K_train)
A = find_N_star(args, bal_tol = bal_tol)
N = max(A) + 1
N_trials = 100
List = []
for _ in range(N_trials):
    K_N =  kernel_estimate(K_train, N)
    y_pred_N = np.sign( clf.decision_function(K_N).flatten())
    List.append(0.5*np.mean(np.abs(y_pred_N - y_train.flatten())))

List = np.sort(List)
print(List[90])
