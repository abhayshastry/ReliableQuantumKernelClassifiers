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
        if scaled:
            X = StandardScaler().fit(X).transform(X)
        return X,y
    if filename=="make_circles":
        X,y = make_circles(200,random_state=0)
        y=2*y - 1
        if scaled:
            X = StandardScaler().fit(X).transform(X)
        return X,y
    infile= open(filename,'rb')
    data = pickle.load(infile)
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
        SV = nominal_classifier.support_
        beta[SV] = nominal_classifier.dual_coef_
        b = nominal_classifier.intercept_
    if svm_type == "primal":
        beta,b,SV = mu.primal_rob_no_dummy(K_train, y_train, delta_1 = 1, delta_2 = 1, shots = 1, C = C)
    if svm_type == "dual":
        alpha,b,SV = mu.dual_robust_quadratic(K_train, y_train, delta = 1, shots = 1, C = C)
        beta[SV] = alpha[SV] * y_train[SV]
    return beta,b,SV

def run_experiment(split_state,shots_array,dataset="my_checkerboard.csv",kernel="havlicek",C=1,delta_1=0.1,delta_2=0.1):
    """Runs experiment to collect the Reliability data of
            of the classifiers versus N"""
    #Test Train Split and Evaluate Kernel:
    X,y = get_dataset(dataset)
    if dataset == "my_checkerboard.csv":
        X = np.hstack((X,X))
    n_qubits = X.shape[1]
    node = create_device(kernel,n_qubits)
    K_star,K_star_test_train,y_train,y_test = evaluate_kernel(X,y,node,split_state=split_state)#Evaluate kernel in given split
    
    #Parameters needed to initialize the reliability metrics
    k=len(shots_array); N_trials = 200
    M_train = len(y_train);tests=len(y_test)

    #Reliability metrics:
    rely_primal_swap = np.zeros(k); rely_primal_gates = np.zeros(k); rely_nsvm_swap = np.zeros(k); rely_nsvm_gates = np.zeros(k);
    frac_rely_primal_swap = np.zeros(k); frac_rely_primal_gates = np.zeros(k); frac_rely_nsvm_swap = np.zeros(k); frac_rely_nsvm_gates = np.zeros(k);

    #prediction and accuracy arrays
    y_pred_primal_swap = np.zeros((k,N_trials,tests)); y_pred_primal_gates = np.zeros((k,N_trials,tests));
    y_pred_nsvm_swap = np.zeros((k,N_trials,tests)); y_pred_nsvm_gates = np.zeros((k,N_trials,tests));
    accuracy_primal_swap = np.zeros((k,N_trials)); accuracy_primal_gates = np.zeros((k,N_trials));
    accuracy_nsvm_swap = np.zeros((k,N_trials)); accuracy_nsvm_gates = np.zeros((k,N_trials));
    
    for shots_index, shots in enumerate(shots_array):
        start = time.time()

        beta_nsvm,b_nsvm,SV_nsvm = nominal_classifier(K_star,y_train,svm_type="SVC",C=C)

        beta_primal_gates, b_primal_gates, SV_primal_gates = mu.primal_robust_socp(K_star, y_train, delta_1 = delta_1/M_train, delta_2 = delta_2, shots = shots, C = C)
        
        beta_primal_swap, b_primal_swap, SV_primal_swap = mu.primal_robust_socp(K_star, y_train, delta_1 = delta_1/M_train, delta_2 = delta_2, shots = shots, C = C,circuit_type = 'swap')

        #stochastic_kernel_matrix requires the shots argument
        swap_kernel_set = mu.skm(K_star_test_train, shots = shots, N_trials= N_trials, circuit_type = 'swap')
        gates_kernel_set = mu.skm(K_star_test_train, shots = shots,N_trials=N_trials, circuit_type = 'gates')

        for i in range(N_trials):
            #Stochastic Kernels: 
            swap_kernel_test_train = swap_kernel_set[i]
            gates_kernel_test_train = gates_kernel_set[i]

            #Predictions:
            y_pred_primal_swap[shots_index,i,:] = mu.classifier_primal(swap_kernel_test_train, beta_primal_swap, b_primal_swap)    
            accuracy_primal_swap[shots_index,i] = mu.accuracy_score(y_test, y_pred_primal_swap[shots_index,i,:])

            y_pred_primal_gates[shots_index,i,:] = mu.classifier_primal(gates_kernel_test_train, beta_primal_gates, b_primal_gates)    
            accuracy_primal_gates[shots_index,i] = mu.accuracy_score(y_test, y_pred_primal_gates[shots_index,i,:])

            y_pred_nsvm_gates[shots_index,i,:] =  mu.classifier_primal(gates_kernel_test_train, beta_nsvm, b_nsvm) #nominal_svm.predict(gates_kernel_test_train)#  # 
            accuracy_nsvm_gates[shots_index,i] = mu.accuracy_score(y_test, y_pred_nsvm_gates[shots_index,i,:])#Nominal SVM. Uses GATES stochastic kernel matrix.

            y_pred_nsvm_swap[shots_index,i,:] = mu.classifier_primal(swap_kernel_test_train, beta_nsvm, b_nsvm) #nominal_svm.predict(swap_kernel_test_train)  #  
            accuracy_nsvm_swap[shots_index,i] = mu.accuracy_score(y_test, y_pred_nsvm_swap[shots_index,i,:])#Nominal SVM. Uses SWAP stochastic kernel matrix.

    rely_primal_swap[shots_index] = mu.robustness(y_pred_primal_swap[shots_index,:,:], y_test, fraction = 1)
    rely_primal_gates[shots_index] = mu.robustness(y_pred_primal_gates[shots_index,:,:], y_test, fraction = 1)
    rely_nsvm_gates[shots_index] = mu.robustness(y_pred_nsvm_gates[shots_index,:,:],y_test, fraction = 1)
    rely_nsvm_swap[shots_index] = mu.robustness(y_pred_nsvm_swap[shots_index,:,:],y_test, fraction = 1)

    frac_rely_primal_swap[shots_index] = mu.robustness(y_pred_primal_swap[shots_index,:,:], y_test, fraction = 2/3)
    frac_rely_primal_gates[shots_index] = mu.robustness(y_pred_primal_gates[shots_index,:,:], y_test, fraction = 2/3)
    frac_rely_nsvm_gates[shots_index] = mu.robustness(y_pred_nsvm_gates[shots_index,:,:],y_test, fraction = 2/3)
    frac_rely_nsvm_swap[shots_index] = mu.robustness(y_pred_nsvm_swap[shots_index,:,:],y_test, fraction = 2/3)
    
    #accuracy_av_swap = np.mean(accuracy_nsvm_swap,axis=1)
    #accuracy_av_gates= np.mean(accuracy_nsvm_gates,axis=1)

    fraction=2/3
    accuracy = [accuracy_nsvm_swap, accuracy_nsvm_gates, accuracy_primal_swap, accuracy_primal_gates]
    rob = [rely_nsvm_swap, rely_nsvm_gates, rely_primal_swap, rely_primal_gates]
    frac_rob = [frac_rely_nsvm_swap, frac_rely_nsvm_gates, frac_rely_primal_swap, frac_rely_primal_gates]
    #Pickle the data, accuracies and robustness
    data = {'train_test':(K_star,K_star_test_train,y_train,y_test), 'shots_array': shots_array,'delta':(delta_1,delta_2),'accuracy':accuracy,'rob': (rob,frac_rob)}
    return data

