import numpy as np
from exp_utils import *

def stochastic_root_finding(f, x_range = (10, 1000), N_queries = 20, N_trials = 1000, delta = 0.1, domain_f = "Int"):
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

def test_fn(x, root = 256, rand_str = 0.1):
    if x < root:
        return (root -x)**2/root + rand_str* np.random.randn()
    else:
        return rand_str*np.random.randn()
def test_fn1 (x, root = 256, rand_str = 0.1):
    if x < root or  x > 2*root:
        return (root -x)**2/root + rand_str* np.random.randn()
    else:
        return rand_str*np.random.randn()






