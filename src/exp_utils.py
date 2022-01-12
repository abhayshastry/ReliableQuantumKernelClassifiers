import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from  tqdm  import tqdm
from sklearn.svm  import SVC
from sklearn.datasets import make_moons
from scipy.stats import unitary_group
from matplotlib.pyplot import scatter,show


from scipy.linalg import svdvals
# As PosixPath, probably won't work on Windows
from q_kernels import *
from qml_utils import *
from ksvm import *


def reliability( y_true,  Y_rand_tensor, eta = 1):
    (N_train, m) = Y_rand_tensor.shape
    y_true = y_true.flatten()
    assert y_true.shape == (m,)
    y_err = 0.5*np.abs(y_true - Y_rand_tensor)
    y_err = np.einsum("ij->j", y_err)
    y_err = y_err <= (1 - eta)*N_train
    return np.sum(y_err)/m

def scatter_plotter(X,y):
    ix1 = np.where(abs(y-1) < 1e-5 )[0]
    ix2 = np.where( abs(y  + 1) < 1e-5)[0]
    X0 = np.asarray([x[0] for x in X])
    X1 = np.asarray([x[1] for x in X])
   #  return X0, X1, ix1
    scatter(X0[ix1], X1[ix1], marker = "P")
    scatter(X0[ix2], X1[ix2], marker = "o")
    show()
    return 




def checkerboard(samples, spacing= 0.1, L= 4,spread  =0.01):
    a = np.linspace(0, L*spacing,L)
    XX, YY = np.meshgrid( a,a)
    X = np.zeros((samples,2))
    y =  np.zeros(samples)

    for t in range(samples):
        i =  np.random.choice(L)
        j =  np.random.choice(L)
        X[t,0] = XX[i,j] +  spread*np.random.normal()
        X[t,1] = YY[i,j] +  spread*np.random.normal()
        if i%2 == j%2:
            y[t] = 0
        else:
            y[t] = 1
    return X,y

def symmetric_donuts(num_samples):
    """generate data in two circles, with flipped label regions
    Args:
        num_samples (int): Number of datapoints
        num_test (int): Number of test datapoints
    Returns:
        X (ndarray): datapoints
        y (ndarray): labels
    """
    # the radii are chosen so that data is balanced
    inv_sqrt2 = 1/np.sqrt(2)

    X = []
    X_test = []
    y = []
    y_test = []

    # Generate the training dataset
    x_donut = 1
    i = 0
    while (i<num_samples):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            X.append([x_donut+x[0],x[1]])
            if r_squared < .25:
                y.append(x_donut)
            else:
                y.append(-x_donut)
            # Move over to second donut
            if i==num_samples//2:
                x_donut = -1

    return X, np.array(y)


def N_star(K_train, y_train,  K_test_train, y_test, R_star,
           N_list =  np.arange(5,500, 20) ,  N_trials = 20, delta = 0.1, C = 1,  Training = "Classical"  ):

    df = N_vs_emp_risk(K_train,  y_train, N_list, K_test_train, y_test,
                       C = C, N_trials = N_trials, Training = "Classical")

    high_prob_remp = []
    for N in N_list:
        high_prob_remp.append( high_prob_upper( df.loc[N,:]["Emp_risk"], delta ) )

#    print(np.asarray(high_prob_remp) - R_star)

#    return N_list, np.asarray(high_prob_remp)-R_star
    return root_by_lin_interpolation(N_list, np.asarray(high_prob_remp)-R_star)

"""
def N_star_from_df(df, R_star,
           N_list =  np.arange(5,500, 20) ,  N_trials = 20, delta = 0.1  ):

    high_prob_remp = []
    for N in N_list:
        high_prob_remp.append( high_prob_upper( df.loc[N,:]["Emp_risk"], delta ) )

    return root_by_lin_interpolation(N_list, np.asarray(high_prob_remp)-R_star)
"""

def N_star_from_df(df, R_star, y_train,  N_list = np.linspace(2,4000, 15, dtype=int), N_trials = 20, delta =0.1 ):
    emp_risk = []
    for x  in df.index:
         emp_risk.append( np.mean (df.loc[x]["y_pred"]*y_train < 0 ))
    df["Emp_risk"] = emp_risk

    high_prob_remp = []
    for N in N_list:
        high_prob_remp.append( high_prob_upper( df.loc[N,:]["Emp_risk"], delta ) )

    return root_by_lin_interpolation(N_list, np.asarray(high_prob_remp)-R_star)

def N_vs_emp_risk (K_train, y_train,  N_list, K_test_train, y_test, C =1, N_trials = 10, Training = "Classical", Training_trials = 10, margin = 0 ):
    clf =  SVC(kernel='precomputed', C = C)
    if Training == "Classical":
        clf.fit(K_train, y_train)
    #TODO Implement training with kernels that are measured during the training phase.
    Data_dict =  {}
    for N in tqdm(N_list):
        for jj in range(N_trials):
            y_pred =  clf.decision_function(kernel_estimate(K_test_train,N))
            Data_dict[(N, jj)] = np.mean(y_pred*y_test < 0)

    df = pd.DataFrame(Data_dict.values(), pd.MultiIndex.from_tuples(Data_dict.keys(), names = ["N_measurements", "repeat"]), columns=["Emp_risk"])
    return df

def N_vs_y_pred (K_train, y_train,  N_list, K_test_train, y_test, C =1, N_trials = 10, Training = "Classical", Training_trials = 10 ):
    clf =  SVC(kernel='precomputed', C = C)
    if Training == "Classical":
        clf.fit(K_train, y_train)
    #TODO Implement training with kernels that are measured during the training phase.
    Data_dict =  {}
    for N in tqdm(N_list):
        for jj in range(N_trials):
            y_pred =  clf.decision_function(kernel_estimate(K_test_train,N))
            Data_dict[(N, jj)] =  {"y_pred":y_pred, "K_err":np.max ( np.abs( kernel_estimate(K_test_train,N) - K_test_train).flatten() )}

    df = pd.DataFrame(Data_dict.values(), pd.MultiIndex.from_tuples(Data_dict.keys(), names = ["N_measurements", "repeat"]), columns=["y_pred", "K_err"])
    return df

def N_vs_y_pred_dict (K_train, y_train,  N_list, K_test_train, y_test, C =1, N_trials = 10, Training = "Classical", Training_trials = 10 ):
    clf =  SVC(kernel='precomputed', C = C)
    if Training == "Classical":
        clf.fit(K_train, y_train)
    #TODO Implement training with kernels that are measured during the training phase.
    Data_dict =  {}
    for N in tqdm(N_list):
        for jj in range(N_trials):
            y_pred =  clf.decision_function(kernel_estimate(K_test_train,N))
            Data_dict[(N, jj)] =  y_pred

    return Data_dict

def high_prob_upper(List, delta):
    """
    Given a list this returns the element in that list such that atmost delta fraction of the
    elements of the list is greater than this element.
    """
    A =  np.sort(List)
    ind =  int(np.floor(delta*len(A)))
    assert ind > 1, "Length of List is not large enough for this delta, increase number of empirical trials"
    return A[-ind]


def root_by_lin_interpolation(x_list,y_list):
    """
    Finds the smallest x_val for which y is by constructing a piece-wise linear function from x_list and y_list
    """
    print("N vs Remp-Rstar", flush=True)
    print(x_list)
    print(y_list)
    begin_sign =  np.sign(y_list)[0]
    if sum(abs( y_list  )) == 0.0:
        return 1.0
    j1 =  np.asarray( np.sign(y_list) == begin_sign  ).nonzero()[0][-1]
    if sum( np.sign(y_list) != begin_sign  ) == 0:
        if begin_sign == 1:
            return x_list[-1]
        if begin_sign == -1:
            return x_list[0]


    j2 =  np.asarray( np.sign(y_list) != begin_sign  ).nonzero()[0][0]
    x1 = x_list[j1]
    x2 = x_list[j2]
    f1,  f2 =  (y_list[j1], y_list[j2])
    x1,  x2 =  (x_list[j1], x_list[j2])
    return x1 - f1* (( x1 - x2)/ (f1 - f2))


def  find_optimal_C (K, y , C_range = "default", cross_val_frac = 0.2, N_trials = 5 ):
    m = len(y)
    m_test = np.int(np.ceil(cross_val_frac*m))
    m_train = m - m_test
    if C_range == "default":
        C_range = np.linspace(1/m , 10 ,50)
    acc_vals = np.zeros(len(C_range))


    for _ in range(N_trials):
        ind = np.random.permutation(m)
        K = K[ind, :][:,ind]
        y = y[ind]
        K_train = K[m_test:m, m_test:m]
        K_test_train = K[:m_test, m_test:m]
        y_train = y[m_test:m]
        y_test = y[:m_test]

        for ii, C in enumerate(C_range):
            clf =  SVC(kernel='precomputed', C = C)
            clf.fit(K_train, y_train)
            Z = clf.decision_function(K_test_train)
            acc_vals[ii] += np.mean(Z*y_test > 0)

    return C_range[np.argmax(acc_vals)]

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
        assert n_qubits > 1
        Z_gate = np.asarray([[1,0],[0,-1]])
        mat_list = [Z_gate]*2
        parity = np.kron(*mat_list)
        if n_qubits > 2:
            for _ in range(n_qubits-2):
                parity =  np.kron(parity, Z_gate)

        V =  unitary_group.rvs(2**n_qubits, random_state = seed)
        M = np.conj(V.T)@ parity @ V
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


    if (key[1] == "Two_Moons") or (key[1] == "Checkerboard") or (key[1])=="SymDonuts":
        if key[1] == "Two_Moons":
            X, y =  make_moons(m)
            X = [X[i,:] for i in range(m)]
            y = 2*y - 1
        if key[1] == "Checkerboard":
            X,y = checkerboard(m)
            X = [X[i,:] for i in range(m)]
            y = 2*y - 1

        if key[1] == "SymDonuts":
            X,y =symmetric_donuts(m)

        X =  StandardScaler().fit(X).transform(X)
        kernel_strings = key[0].split(",")
        if len(kernel_strings) == 1:
            n_qubits = 2
        else:
            reps = int(kernel_strings[1])
            n_qubits = 2*reps
            X = [np.hstack([x]*reps) for x in X]

        if kernel_strings[0] == "Angle":
            print(f"Generating data:{key}")
            node = device_wrapper(n_qubits, angle_kernel)
            K= quantum_kernel_matrix(X, node, weights = None)

        if  kernel_strings[0]== "Havliscek":
            print(f"Generating data:{key}")
            node = device_wrapper(n_qubits, havlicek_kernel)
            K= quantum_kernel_matrix(X, node, weights = None)

        if  kernel_strings[0]== "Circ-Hubr":
            print(f"Generating data:{key}")
            qc, params = parametric_quantum_kernel_1(n_qubits)
            W =  [q_features_from_circ(x, qc, params) for x in X]
            W = np.hstack(W)
            K =  np.dot(W.T,  np.conj(W))
            K = np.real(np.einsum("ij, ij  -> ij", K,np.conj(K)))

        if  kernel_strings[0]== "QAOA":
            L = 2
            wn = 3  if n_qubits == 2 else 2*n_qubits
            weights = np.random.rand(L, wn)
            node = device_wrapper(n_qubits, qaoa_kernel)
            K= quantum_kernel_matrix(X, node, weights = weights)


    return K,y

def eta_from_N(K, n_meas ,delta, N_trials = 100, test = "swap"):
    """
    Find eta  such that K_n - K < eta
    """
    estimates_list = []
    for _ in range(N_trials):
        estimates_list.append(svdvals(kernel_estimate(K, n_meas) - K)[0])
    upper = high_prob_upper(estimates_list, delta)
    return upper


def N_from_eta(K, eta, delta = 0.1, N_trials = 20, N_list = np.linspace(2,5000,10)):
    eta_list = []
    for N in N_list:
        eta_list.append( eta_from_N(K,N, delta, N_trials = N_trials))
    return root_by_lin_interpolation(N_list, np.asarray(eta_list)-eta)
