import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
import pickle
import utils.qml_utils as qu
import utils.math_utils as mu
import time
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=3)


def get_dataset(filename,scaled=True):
    if filename=="make_moons":
        X,y = make_moons(200,random_state=0)
        y=2*y-1
    if filename=="make_circles":
        X,y = make_circles(200,random_state=0)
        y=2*y - 1
    if filename =="make_circles" or "make_moons":
        if scaled:
            X= StandardScaler().fit(X).transform(X)
        return X,y
    data = pickle.load(filename,'rb')
    X=data['X']
    y=data['y']
    if scaled:
        X = StandardScaler().fit(X).transform(X)
    return X,y

def create_device(kernel_name,n_qubits):
    if kernel_name == "havlicek":
        quantum_kernel = qu.havlicek_kernel
    if kernel_name == "qaoa":
        quantum_kernel = qu.qaoa
    if kernel_name == "angle":
        quantum_kernel = qu.angle_kernel
    #Make device:
    node = qu.device_wrapper(n_qubits, quantum_kernel)
    return node

def evaluate_kernel(X,y,node,split_state=0,train_size=60,n_repeats=1,params=None,sigma=1):
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size,random_state=split_state)
    if node== "gaussian":
        start=time.time()
        K_star = mu.kernel_matrix(X_train,kernel_fn=mu.gauss_kernel,sigma=sigma)
        K_star_test_train=mu.test_train_kernel(X_train,X_test,kernel_fn=mu.gauss_kernel,sigma=sigma)
        end=time.time()
        print("[Gaussian case] Total time to evaluate both train and test-train kernel:",end-start)
        return K_star, K_star_test_train,y_train,y_test
    start = time.time()
    K_star = qu.quantum_kernel_matrix(X_train,  node, weights = params,n_repeats=n_repeats)
    end = time.time()
    assert K_star.shape[0]==train_size, "Train size is "
    print("[Quantum case] Time taken for evaluating training kernel:",end-start)
    K_star_test_train = qu.quantum_test_train_kernel(X_test,X_train, quantum_node = node, weights = params,n_repeats=n_repeats)
    end2 = time.time()
    print("[Quantum Case] Time taken for evaluating test-train kernel:",end2-end)
    return K_star,K_star_test_train,y_train,y_test

def nominal_classifier(K_train,y_train,svm_type="primal",C=1):
    #Return beta and b. beta to be of length y_train and not the support.
    M_train= len(y_train)
    beta= np.zeros(M_train)
    if svm_type=="SVC":
        nominal_classifier = SVC(kernel="precomputed",C=C).fit(K_train,y_train)
        alpha_iy_i = nominal_classifier.dual_coef_
        b = nominal_classifier.intercept_
        SV = nominal_classifier.support_
        y = y_train[SV]
        beta[SV] = y*alpha_iy_i
    if svm_type == "primal":
        beta,b,SV = mu.primal_robust_socp(K_train, y_train, delta_1 = 1, delta_2 = 1, shots = 1, C = C)
    if svm_type == "dual":
        alpha,b,SV = mu.dual_robust_quadratic(K_train, y_train, delta = 1, shots = 1, C = C)
        beta[SV] = alpha[SV] * y_train[SV]
    return beta,b,SV


