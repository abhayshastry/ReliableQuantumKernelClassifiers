import numpy as np
from scipy.linalg import norm, eigh
import matplotlib.pyplot as plt
import cvxpy as cp


#def Ber(p, N):
#    samp =  np.random.rand(N)
#    return [1 if s<p else 0 for s in samp ]

def is_positive(x):
    """
    Check if a vecor lies in R+
    """
    for xi  in list(x):
        if xi <= 0.0:
            return False
    return True
    
def gauss_kernel(v,u, sigma = 1.0):
    return np.exp(- norm(np.asarray(v) - np.asarray(u))**2/(sigma*sigma))

def kernel_matrix(x_list, kernel_fn =  np.dot):
    M =  len(x_list)
    K =  np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            K[i,j] =  kernel_fn(x_list[i], x_list[j])
            K[j,i] = K[i,j]
    return K

def kernel_estimate(K, n_meas, ishermitian = True):
    M,N = K.shape
    ishermitian =  np.allclose(K, K.T, rtol=1e-10, atol=1e-10)
    K_bar =  np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            #samples =  Ber( 0.5*(1 - K[i,j]), n_meas )
            try:
                K_bar[i,j] = 1 -  (2/n_meas)* np.random.binomial(n_meas, 0.5*(1 - K[i,j]))
            except ValueError:
                if abs(0.5*(1 - K[i,j])) < 1e-15:
                    K_bar[i,j] = 1.0
                else:
                    print( "p is ", 0.5*(1 - K[i,j]) )
            if ishermitian:
                K_bar[j,i] = K_bar[i,j]
    return K_bar

def test_train_kernel(test_array, train_array, kernel_fn= np.dot):
    """
    Retruns a n_test * n_train matrix with the kernel entries for testing

    """
    s1 =  test_array.shape
    s2 =  train_array.shape
    assert s1[1] == s2[1]
    K =  np.zeros((s1[0], s2[0]))
    for i in range(s1[0]):
        for j in range(s2[0]):
            K[i,j] = kernel_fn(test_array[i,:], train_array[j,:])

    return  K



def ksvm_opt(K, y_list):
    """
    solves the kernel-svm optimization problem

    """
    M =  len(y_list)
    assert K.shape == (M,M)
    Y =  np.diag(y_list)
    y_vec =  np.asarray(y_list).reshape(1,M)
    B = Y@K@Y

    t =  cp.Variable()
    alpha =  cp.Variable((M,1))
    e =  np.ones((1,M))

    obj =  cp.Minimize( 0.5*t - e@alpha )
    constraints = [ 0 <= alpha, y_vec@alpha == 0,  cp.quad_form(alpha, B) <= t ]
    problem =  cp.Problem(obj, constraints)
    print("Optimal value", problem.solve())

    return alpha.value

def ksvm_line_2D(x_list, y_list, K, thresh =  1e-3):
    
    M =  len(y_list)
    sv =  ksvm_opt(K, y_list)
    sv_max =  np.max( np.abs(sv)  )
    sv_index_list = [ i for i in range(M) if abs(sv[i])/sv_max >  thresh ]
    w  = sum([sv[i][0]*y_list[i]*x_list[i] for i in range(M)])
    b1 =  sum([y_list[i] for i in sv_index_list])
    b2 =  sum([ y_list[i]*sv[i]*K[i,j] for i in sv_index_list for j in sv_index_list ])[0]
    b =  (1.0/len(sv_index_list))*(b1 -b2)
    y = lambda x:  (-b - w[0]*x)/w[1]
    return y, w,  b

def ksvm_classifier(x_list,  y_list,K,  kf = np.dot , thresh =  1e-3):
    M =  len(y_list)
    sv =  ksvm_opt(K, y_list)
    sv_max =  np.max( np.abs(sv)  )
    sv_index_list = [ i for i in range(M) if abs(sv[i])/sv_max >  thresh ]
    b1 =  sum([y_list[i] for i in sv_index_list])
    b2 =  sum([ y_list[i]*sv[i]*K[i,j] for i in sv_index_list for j in sv_index_list ])[0]
    b =  (1.0/len(sv_index_list))*(b1-b2)
    classifier =  lambda x:   sum([sv[i][0]*y_list[i]*kf(x_list[i],x) for i in sv_index_list] + b)
    return classifier
