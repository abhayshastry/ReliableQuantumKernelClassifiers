import numpy as np
import cvxpy as cp
#from cvxopt import matrix, solvers
from scipy.linalg import sqrtm, norm
from sklearn.svm import SVC
import warnings
from cvxpy.atoms.affine.wraps import psd_wrap

def kernel_matrix_decision_function(test_train_kernel, alpha, y, intercept_):
#Note: dual_coef_ is array product alpha*y_train. dual_coef_ is a paramater for SVC, svc.dual_coef_ length is equal to support
    N_test = test_train_kernel.shape[0]
    N_train = test_train_kernel.shape[1]
    assert N_train == len(y), "N_train and y_train are mismatched"
    assert N_train == len(alpha), "N_train and alpha coefficients are mismatched"
    F = np.zeros(N_test)
    dual_coef_ = alpha*y
    for l_test in range(N_test):
        K = test_train_kernel[l_test,:]#Array of length N_train
        F[l_test] = np.dot(dual_coef_,K) + intercept_
    return F

def kernel_matrix_classifier(test_train_kernel, alpha, y, intercept_):
    return np.sign(kernel_matrix_decision_function(test_train_kernel, alpha, y, intercept_))

def accuracy_score(x,y):
    A = x-y
    return 1 - (np.count_nonzero(A)/len(A))


def eigmin(K):
    return np.amin(np.linalg.eigvalsh(K))

def eigsum(K):
    return np.sum(np.linalg.eigvalsh(K))

def isPSD(K, tol = 10**-16):
    if eigmin(K)<-tol:
        return False
    else:
        return True


def tik_reg(K): #Tikhanov regularization as discussed in Hubregsten2021
    if eigmin(K)<0:
#        print('tik-reg')
        return K - eigmin(K)*np.identity(len(K))
    else:
        return K

def threshold_reg(K): #Thresholding regularization as discussed in Hubregten2021
    w, v = np.linalg.eigh(K)
    w = np.maximum(w,np.zeros(len(w)))
    return np.matmul(v,np.matmul(np.diag(w),v.T))

def estimate_kappa(N, epsilon_tol = 0.1):#Estimated kappa to ensure PSD of Kernel
    kappa = np.sqrt(1/N) * np.sqrt(1-epsilon_tol/epsilon_tol)
    return kappa

#def robust_reg(K,N, epsilon_tol = 0.1): #Robustness lemma from Abhijith
#    kappa = np.sqrt(1/N)* np.sqrt(1/epsilon_tol)
#    return K + kappa*np.identity(len(K))

def gauss_kernel(v,u, sigma = 1.0):
    return np.exp(- norm(np.asarray(v) - np.asarray(u))**2/(sigma*sigma))


def kernel_matrix(X, kernel_fn = np.dot, sigma = None):
    L = len(X)
    KernelMatrix = np.zeros((L,L))
    if sigma == None:
        for l in range(L):
            for m in range(l):
                KernelMatrix[l,m] = kernel_fn(X[l], X[m])
    else:
        for l in range(L):
            for m in range(l):
                KernelMatrix[l,m] = kernel_fn(X[l], X[m], sigma)
    return KernelMatrix + KernelMatrix.T + np.identity(L)

def stochastic_kernel_matrix(exact_kernel_matrix, shots = 0, circuit_type = 'gates'):# Use this instead of kernel_estimate
    if shots == 0:
        return exact_kernel_matrix
    if circuit_type == 'gates':
        return np.random.binomial(shots, exact_kernel_matrix)/shots
    if circuit_type == 'swap':
        Y = np.random.binomial( shots, (1-exact_kernel_matrix)/2)/shots
        return 1 - 2*Y

def skm(exact_kernel_matrix, shots = 0, circuit_type = 'gates', N_trials = 1):# Use this instead of kernel_estimate
    (r,c) = exact_kernel_matrix.shape
    if shots == 0:
        return exact_kernel_matrix
    if circuit_type == 'gates':
        return np.random.binomial(shots, exact_kernel_matrix, (N_trials, r,c))/shots
    if circuit_type == 'swap':
        Y = np.random.binomial( shots, (1-exact_kernel_matrix)/2, (N_trials, r, c))/shots
        return 1 - 2*Y

def skm2(exact_kernel_matrix, shots = 0, circuit_type = 'gates', N_trials = 1):# Use this instead of kernel_estimate
    (r,c) = exact_kernel_matrix.shape
    if shots == 0:
        return exact_kernel_matrix
    gates_set = np.zeros((N_trials,r,c))
    swap_set = np.zeros((N_trials,r,c))
    for i in range(N_trials):
        gates_set[i] = np.random.binomial(shots, exact_kernel_matrix)/shots
        Y_swap_set = np.random.binomial( shots, (1-exact_kernel_matrix)/2)/shots
        swap_set[i] = 1 - 2*Y_swap_set
    if circuit_type=='gates':
        return gates_set
    if circuit_type == 'swap':
        return swap_set

def test_train_kernel(X_test, X_train, kernel_fn = np.dot, sigma = None):#Output must be dim (L_test, L_train)
    L_test = len(X_test)
    L_train = len(X_train)
    TestTrainKernel = np.zeros((L_test,L_train))
    for l_test in range(L_test):
        for l_train in range(L_train):
            TestTrainKernel[l_test,l_train] = kernel_fn(X_test[l_test],X_train[l_train])
    if sigma!=None:   
        for l_test in range(L_test):
            for l_train in range(L_train):
                TestTrainKernel[l_test,l_train] = kernel_fn(X_test[l_test],X_train[l_train], sigma)
    return TestTrainKernel

def beta(K, circuit_type = 'gates'):
    n = K.shape[0]
    if circuit_type == 'gates':
        beta = K - K*K
    if circuit_type == 'swap':
        beta = np.ones((n,n)) - K*K
    return beta

def beta_subgauss(K, circuit_type = 'gates'):
    n = K.shape[0]
    beta_sg = np.zeros((n,n))
    if circuit_type == 'gates':
        p = K
        Prefactor = 1
    if circuit_type == 'swap':
        p = 0.5 * (1-K)
        Prefactor = 4
    q = 1 - p
    for i in range(n):
        for j in range(i):
            beta_sg[i,j] =0.5* (p[i,j] - q[i,j])/(np.log(p[i,j])- np.log(q[i,j]))

    return Prefactor*(beta_sg + beta_sg.T)



def rsvm_socp(K, beta, y_list, kappa=0, C  = np.inf, sv_threshold =  1e-3):
    """
    K = Precomputed kernel
    y_list =  list of binary labels of the training data
    kappa = tunable robustness parameter. Setting to zero should give same results as svc
    C = bound on the maximum value of alphas
    sv_threshold = points whose normalized alpha value is less than this is not counted as a support vector

    returns: alpha, b, SV
    alpha = optimal value of the dual variables
    b =  constant term in the decision function
    SV =  index of the support vectors

    """
    n = len(y_list)
    assert K.shape ==  (n,n), "Dimensions of K mismatched with y"
    assert beta.shape == (n,n), "Dimensions of beta mismatched with y"
    assert isPSD(K), "K is not PSD"
    assert isPSD(beta),"beta is not PSD"
    sqrt_beta =  np.real(sqrtm(beta))
    #print(sqrt_beta)
    sqrt_K =  np.real(sqrtm(K))
    alpha,t =   robust_svm_soc(sqrt_K, sqrt_beta, np.diag(y_list), kappa, C = C)
    scaled_alpha = alpha/np.max(alpha)
    SV =  [i  for i in range(n) if scaled_alpha[i] > sv_threshold ]    
    assert len(SV) >0, "Support vectors empty. Tune the threshold"
    b = 0
    for i in SV:
        s = y_list[i]
        for j in SV:
            s +=  -1*alpha[j]*y_list[j]*K[i,j]
        b += s
    return alpha, b/len(SV), SV

def robust_svm_soc(sqrt_K, sqrt_beta, Y, kappa, C = np.inf):
    n =  sqrt_K.shape[0]
    assert sqrt_beta.shape == sqrt_K.shape
    assert Y.shape == sqrt_K.shape
    N  = 2*n + 2
    x = cp.Variable(N)
    ##x[:n] =  \alpha, x[n:2n]= =  \nu, x[2n] =  t, x[2n+1] =  t^\prime
    A = []
    b = []
    c = []
    d = []

    ###The first constraint

    A1 =  np.zeros((N,N))
    A1[n:2*n, n:2*n] =  kappa*sqrt_beta
    b1 = np.zeros(N)
    c1 = np.zeros(N)
    c1[2*n] = 1
    c1[2*n + 1] = -1
    d1 = 0

    A.append(A1)
    b.append(b1)
    c.append(c1)
    d.append(d1)
    
    #Second constraint
    A2 =  np.zeros((N,N))
    A2[:n, :n] =  2*sqrt_K@Y
    A2[N-1,N-1] = 1
    b2 =  np.zeros(N)
    b2[N-1] = -1
    c2 =  np.zeros(N)
    c2[N-1] = 1
    d2 = 1
    

    A.append(A2)
    b.append(b2)
    c.append(c2)
    d.append(d2)
        
        
    e =  np.ones(n)
    for i in range(n):
        P =  np.zeros((n,n))
        P[i,i] = 1
        A3i =  np.zeros((N,N))
        A3i[:n,:n] =  2*P[:,:]
        A3i[n:2*n, n:2*n] =  P[:,:]
        b3i =  np.zeros(N)
        b3i[n+i] = -1
        c3i =  np.zeros(N)
        c3i[n+i] = 1
        d3i = 1

        A.append(A3i[:,:])
        b.append(b3i[:])
        c.append(c3i[:])
        d.append(d3i)
    assert len(A) ==  n+2

    soc_constraints = [
      cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(n+2)]

    linear_constraints = [x[i] >= 0  for i in range(n) ] 
    if C !=np.inf:
        linear_constraints +=   [x[i] <= C  for i in range(n) ] 
     
    temp =  np.zeros(N)
    temp[:n] = Y@np.ones(n)
    linear_constraints.append(x.T @ temp == 0)
    e =  np.zeros(N)
    e[:n] =  -1*np.ones(n)
    e[N-2] =  0.5
    prob = cp.Problem(cp.Minimize(e.T@x),
                  soc_constraints + linear_constraints)
#    print('problem being solved. prob.solve() is',prob.solve( ))
    prob.solve()
    return x[:n].value, x[2*n].value

def robustness(y_pred, y_test, fraction = 1):#y_pred is predicted labels, y_test are true labels
    N_trials = y_pred.shape[0]
    M_test = y_pred.shape[1]
    assert M_test == len(y_test), "y_pred should be of form N_trials X M_test"
    robust = np.zeros(M_test) 
    for i in range(M_test):
        count = 0
        for k in range(N_trials):
            if y_pred[k,i] == y_test[i]:
                count += 1
        if count/N_trials >= fraction:
            robust[i] = 1
    return sum(robust)/M_test
    
#def optimal_kernel_correction(A):
    """
    A  should be K^2 for the SWAP test
    
    """
#    n =  A.shape[1]
#    x = cp.Variable(n)
#    e =  np.ones(n)
#    prob =  cp.Problem(cp.Minimize(cp.quad_form(x, A)), 
#                                   [x >= 0, e@x == 1])
#   prob.solve()
#   return prob.value

def shifted_ksvm(K,y, shift = 0, C =  np.inf, svc_intercept = False):#RSVM_QP is how it is named in the text
    n = len(y)
    K_shifted = K + shift*np.eye(n)
    if ~isPSD(K_shifted):
        K_shifted = threshold_reg(K_shifted)
    clf = SVC(kernel='precomputed'  ,C = C)
    clf.fit(K_shifted, y)
    alpha_y =  clf.dual_coef_
    alpha = np.zeros(len(y))
    SV =  clf.support_
    b = 0
    for p,i in enumerate(SV):
        alpha[i] =  alpha_y[0,p]*y[i]
        s = y[i]
        for t,j in enumerate(SV):
            s +=  -1*alpha_y[0,t]*K[i,j]
        b += s
    if svc_intercept:
        b = clf.intercept
    return alpha, b/len(SV), SV

def eigmax(K):
    return np.amax(np.linalg.eigvalsh(K))

def opnorm(K):
    return np.amax(np.abs(np.linalg.eigvalsh(K)))

def simplex_shift(K, kappa = 1):
    
    n = K.shape[1]
    x = cp.Variable(n)
    l = cp.Problem(cp.Minimize(cp.quad_form(x,K*K)), [x>=0, cp.sum(x) == 1])
    l.solve()
    l_star = l.value

    shift = kappa*np.sqrt(1-l_star)
    return shift

#New Formulation
def kappa(delta):
    return np.sqrt(2*np.log(1/delta))

def primal_robust_socp(K_star, y_train, delta_1 = 0.1, delta_2 = 0.1, C = 1, shots = 20, circuit_type = 'gates',sv_threshold=1e-3,numerical_shift=1e-6):
    n = K_star.shape[1]
    N = 2*n + 2
    K_star = K_star + numerical_shift*np.eye(n)
    cp.settings.EIGVAL_TOL = 1e-6
    g = 1
    if circuit_type == 'swap':
        g = 2
    x = cp.Variable(N)
    #The SOCP Constraint:
    A1 = np.zeros((N,N))
    b1 = np.zeros(N)
    c1 = np.zeros(N)
    d1 = 0
    A1[n:2*n,n:2*n] = np.eye(n)
    c1[N-1] = 1
    socp_constraint = [cp.SOC(c1.T@x+d1,A1@x+b1)]
    #Linear Constraints
    linear_constraint_1 = [x[i]>=0 for i in range(n)]
    Ai = np.zeros((n,N))
    Ai[:n,:n] = np.eye(n)
    Ai[:n,n:2*n]=np.multiply(y_train,K_star).T#np.diag(y_train)@K_star#
    Ai[:n,N-2] = y_train
    Ai[:n,N-1] = -kappa(delta_1)*(g/np.sqrt(shots))*np.ones(n)
        #linear_constraints += x[i]>= 1 - y_train[i]*(np.dot(x[n:2*n],K_star[:,i]) + x[2*n]) + kappa(delta_1)*(g/np.sqrt(shots))*x[N-1]#cannot be passed this way. See below.
    linear_constraint_2= [Ai@x>=np.ones(n)]
    e = np.zeros(N)
    e[:n]= np.ones(n)
    K_shifted = K_star + kappa(delta_2)*(g/np.sqrt(shots))* np.eye(n)
    Q= np.zeros((N,N))
    Q[n:2*n,n:2*n] = K_shifted
    Q = psd_wrap(Q)
    h_rob = cp.Problem(cp.Minimize(C*e.T@x + 0.5* cp.quad_form(x,Q)) ,socp_constraint + linear_constraint_1 + linear_constraint_2)
    h_rob.solve(solver=cp.CVXOPT)
    primal_optimal_value = h_rob.value
    b = x[2*n].value
    beta = x[n:2*n].value #VECTOR. In previous formalism, symbol beta was used for nXn matrix
    scaled_beta = abs(beta)/np.max(abs(beta))
    SV =  [i  for i in range(n) if scaled_beta[i] > sv_threshold ]   #Indices of support vectors 
    assert len(SV) >0, "Support vectors empty. Tune the threshold"
    return beta, b, SV# primal_optimal_value

def dual_robust_quadratic(K_star, y_train, delta=0.1, C = np.inf, shots = 20, circuit_type = 'gates',sv_threshold = 1e-4):
    g = 1
    n= K_star.shape[1]
    if circuit_type == 'swap':
        g = 2
    x = cp.Variable(n)
    e = np.ones(n)
    K_rob = K_star - (g*kappa(delta)/2/np.sqrt(shots)) * np.eye(n)
    if ~isPSD(K_rob):
        warnings.warn('K_rob is not Positive. Solving after making it positive via threshold regularization.')
    K_rob_reg = threshold_reg(K_rob)
    linear_constraint_1 = [x[i]>=0 for i in range(n)]
    linear_constraint_2 = [x[i]<=C for i in range(n)]
    linear_constraint_3 = [y_train.T@x==0]
    Y = np.diag(y_train)
    Q = Y@K_rob_reg@Y
    h_rob_dual = cp.Problem(cp.Maximize(e.T@x - 0.5*cp.quad_form(x,Q)), linear_constraint_1 + linear_constraint_2 + linear_constraint_3)
    h_rob_dual.solve()
    alpha = x.value
    dual_optimal_value = h_rob_dual.value
    #Finding intercept b:
    scaled_alpha = alpha/np.max(alpha)
    SV =  [i  for i in range(n) if scaled_alpha[i] > sv_threshold ]   #Indices of support vectors 
    assert len(SV) >0, "Support vectors empty. Tune the threshold"
    b = 0
    for i in SV:
        s = y_train[i]
        for j in SV:
            s +=  -1*alpha[j]*y_train[j]*K_rob_reg[i,j]
        b += s
    return alpha, b/len(SV), SV#, dual_optimal_value

def primal_robust_socp_no_dummy(K_star, y_train, delta_1 = 0.1, delta_2 = 0.1, 
                        C = 1, shots = 20, circuit_type = 'gates', 
                        sv_threshold = 1e-3, numerical_shift = 1e-6): 
    g = 1
    if circuit_type == 'swap':
        g = 2
        
    cp.settings.EIGVAL_TOL = 1e-06
    n = K_star.shape[1]
    ##Shifting the kernel to ensure numerical stability of the solver
    K_star = K_star + numerical_shift*np.eye(n)
    Y=  y_train.reshape(n,1) 
    beta = cp.Variable((n,1))
    v = cp.Variable()
    t = cp.Variable()
    k1 = cp.Parameter(nonneg=True)
    k2 = cp.Parameter(nonneg=True)

    loss = cp.sum(cp.pos(1 + k1*t - cp.multiply(Y, K_star @ beta + v)))

    socp_constraint = [cp.SOC(t,beta)]
    reg1 = cp.quad_form(beta,psd_wrap(K_star) )
    #reg2 = cp.norm(beta, 2)
    reg2 = cp.quad_form(beta,psd_wrap(np.eye(n)))
    prob = cp.Problem(cp.Minimize(C*loss + 0.5*reg1 + 0.5*k2* reg2  ), socp_constraint)
    k1.value = kappa(delta_1)*(g/np.sqrt(shots))
    k2.value = kappa(delta_2)*(g/np.sqrt(shots))
    prob.solve(solver=cp.CVXOPT)
    scaled_beta = np.abs(beta.value)/np.max(np.abs(beta.value))
    SV =  [i  for i in range(n) if scaled_beta[i] > sv_threshold ]
    return beta.value.flatten(), v.value, SV


def classifier_primal(test_train_kernel, beta, b):
    N_test = test_train_kernel.shape[0]
    N_train = test_train_kernel.shape[1]
    assert N_train == len(beta), "N_train and beta are mismatched"
    F = np.zeros(N_test)
    for l_test in range(N_test):
        K = test_train_kernel[l_test,:]
        F[l_test] = np.dot(beta,K) + b
    return np.sign(F)

def classifier_dual(test_train_kernel, alpha, y, b,SV):
    return kernel_matrix_classifier(test_train_kernel[:,SV],alpha[SV],y[SV],b)
