import numpy as np
import utils.math_utils as mu
import utils.qml_utils as qu

from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time
import pickle
import sys
import matplotlib.pyplot as plt
#Data and preprocessing

infile = open('my_moons.csv', 'rb')
data = pickle.load(infile)
X,y= data['X'], data['y']   

#X,y=load_iris(return_X_y=True) #make_moons(n_samples=100)
#X = X[50:150]
#y = y[50:150]
print('Mean and Variance of X is', np.mean(X), 'and', np.var(X), 'respectively')
#print('X looks like this: ', X)
#X = StandardScaler().fit(X).transform(X)
#y = 2*y-3 # For Iris if X= X[50:150]
y = 2*y-1
print('y looks like this: ', y)
print('Mean and Variance of X is (after rescaling)', np.mean(X), 'and', np.var(X), 'respectively')
#Test Train Split
random_state = None

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.2, random_state = random_state)#Save the random_state while pickling
#print('X_train looks like ', X_train)
#print('y_train looks like ', y_train)
#sys.exit()
#Print floats to 3rd decimal only
np.set_printoptions(precision=3)

#Create the Quantum Kernel Device with the specified kernel
n_qubit = X.shape[1]
#Some parameters for QAOA Embedding (Lloyd). For n_qubits = 4
#QAOA Section deleted

print(n_qubit,'num qubits')
#quantum_kernel = qu.qaoa_kernel # Lloyd Kernel
quantum_kernel = qu.havlicek_kernel #Havlicek Kernel
if quantum_kernel == qu.havlicek_kernel:
    params = None
node = qu.device_wrapper(n_qubit, quantum_kernel) #Creates a node with qubits = n_qubit and the chosen kernel
kernel_evaluation1 = node(X_train[0],X_train[1],n_qubit,weights= params)
#print('sample kernel evaluation K(X[0],X[1]):', kernel_evaluation1)
print(type(kernel_evaluation1))

print('Properties of the device can be accessed via node.device...')
print('Number of qubits from the device is ', len(node.device.wires))

start = time.time()
n_repeats = 2#Repeats of the basic Embedding in the Havlicek Kernel
#K_star = qu.quantum_kernel_matrix(X_train,  node, weights = params)
end = time.time()
print('K_train time taken is ', end-start)
#delta=1 for cases with no randomness
delta_1 = 1
delta_2 = 1
delta_dual = 1
C = 1000
K_gauss = mu.kernel_matrix(X_train,kernel_fn=mu.gauss_kernel,sigma=0.5)
#print(K_gauss); K_star = K_gauss
K_star = qu.quantum_kernel_matrix(X_train, node, weights = params)
print('minimum eigenvalue of K_gauss:',mu.eigmin(K_gauss))
print('minimum eigenvalue of K_quantum:',mu.eigmin(K_star))

#Nominal SVM using K_star. Predictions using K_star_test_train. No randomness.
nominal_svm = SVC(kernel = 'precomputed', C = C).fit(K_star, y_train)
#K_star_test_train = qu.quantum_test_train_kernel(X_test,X_train, quantum_node = node, weights = params)
K_star_test_train = mu.test_train_kernel(X_test,X_train, kernel_fn = mu.gauss_kernel, sigma = 0.5)
y_pred_nsvm_gates = nominal_svm.predict(K_star_test_train)
print('Accuracy score for Exact SVM  ',mu.accuracy_score(y_test,y_pred_nsvm_gates))

#NOTE: beta designates the variable in the Primal Problem. Not the matrix related to K_star, as done in the previous formulation.

beta,b,opt_value = mu.primal_robust_socp(K_gauss, y_train, delta_1 = delta_1, delta_2 = delta_2, shots = 100, C =C)
alpha,b_dual, SV, dual_opt_value = mu.dual_robust_quadratic(K_gauss, y_train, delta = delta_dual, shots = 100, C = C)
print('The optimal value of the function h_rob is', opt_value)
print('Intercept b in primal is:', b, 'and in dual is:', b_dual,'From the SVC:',nominal_svm.intercept_)
alpha_abs = abs(alpha)
print('The betas are :', beta)
print('The primal alpha[i]*y[i] are:',[x*y_train[i] for i,x in enumerate(alpha_abs) if x>=1e-10] )
print('The y_i*alpha_i coming from the sklearn SVC is: ', nominal_svm.dual_coef_)
print('Support vectors in the Dual:', SV)
print('Support vectors in Primal:', [i for i,x in enumerate(alpha_abs) if x>=1e-10])
y_pred_primal = mu.classifier_primal(K_star_test_train, beta, b)
y_pred_dual = mu.classifier_dual(K_star_test_train, alpha, y_train, b,SV)
print('Predicted labels are :', y_pred_primal)
print('Accuracy in the primal formulation:', mu.accuracy_score(y_pred_primal, y_test))
print('Accuracy in the Dual Formulation:', mu.accuracy_score(y_pred_dual,y_test))
print('Agreement with the Dual SVM:', mu.accuracy_score(y_pred_primal,y_pred_dual))
print('The optimal values of primal and dual are: ', opt_value, 'and', dual_opt_value)

#Introduce Stochasticity. N, N_trials. delta_dual = 0.1. delta_1=delta_2=0.05 for the primal
delta_1 = 0.1
delta_2 = 0.1
delta_dual = 0.2

shots_array = [10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,500000]
N_trials = 100; k = len(shots_array); tests = len(y_test)
dual_opt_val_array = np.zeros((k,2))
primal_opt_val_array=np.zeros((k,2))

#Reliability metrics:
rely_primal_swap = np.zeros(k); rely_primal_gates = np.zeros(k); rely_dual_swap = np.zeros(k); rely_dual_gates = np.zeros(k);
frac_rely_primal_swap = np.zeros(k); frac_rely_primal_gates = np.zeros(k); frac_rely_dual_swap = np.zeros(k); frac_rely_dual_gates = np.zeros(k);

#prediction and accuracy arrays
y_pred_primal_swap = np.zeros((k,N_trials,tests)); y_pred_primal_gates = np.zeros((k,N_trials,tests)); y_pred_dual_swap = np.zeros((k,N_trials,tests)); y_pred_dual_gates = np.zeros((k,N_trials,tests));
accuracy_primal_swap = np.zeros((k,N_trials)); accuracy_primal_gates = np.zeros((k,N_trials)); accuracy_dual_swap = np.zeros((k,N_trials)); accuracy_dual_gates = np.zeros((k,N_trials));
#For the nominal SVM. Use only GATES case.
rely_nsvm_gates = np.zeros(k); frac_rely_nsvm_gates=np.zeros(k);
y_pred_nsvm_gates= np.zeros((k,N_trials,tests)); y_pred_primal_swap = np.zeros((k,N_trials,tests));
accuracy_nsvm_gates=np.zeros((k,N_trials)); accuracy_primal_swap= np.zeros((k,N_trials))
rely_nsvm_swap = np.zeros(k); rely_nsvm_gates = np.zeros(k); 
frac_rely_nsvm_swap = np.zeros(k); frac_rely_nsvm_gates = np.zeros(k);
y_pred_nsvm_swap = np.zeros((k,N_trials,tests)); y_pred_nsvm_gates = np.zeros((k,N_trials,tests));
accuracy_nsvm_swap = np.zeros((k,N_trials)); accuracy_nsvm_gates = np.zeros((k,N_trials)) 
M_train = len(y_train)
beta_primal_gates_array=[]
for shots_index, shots in enumerate(shots_array):
    start = time.time()

    beta_primal_gates, b_primal_gates, primal_opt_value_gates = mu.primal_robust_socp(K_star, y_train, delta_1 = delta_1/M_train, delta_2 = delta_2, shots = shots, C = C)
    alpha_dual_gates, b_dual_gates, SV_dual_gates, dual_opt_value_gates = mu.dual_robust_quadratic(K_star, y_train, delta = delta_dual, shots = shots, C = C)
    
    beta_primal_swap, b_primal_swap, primal_opt_value_swap = mu.primal_robust_socp(K_star, y_train, delta_1 = delta_1/M_train, delta_2 = delta_2, shots = shots, C = C,circuit_type = 'swap')
    alpha_dual_swap, b_dual_swap, SV_dual_swap, dual_opt_value_swap = mu.dual_robust_quadratic(K_star, y_train, delta = delta_dual, shots = shots, C = C,circuit_type = 'swap')

    dual_opt_val_array[shots_index,:] = [dual_opt_value_gates, dual_opt_value_swap]
    primal_opt_val_array[shots_index,:]= [primal_opt_value_gates,primal_opt_value_swap]
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

        y_pred_dual_swap[shots_index,i,:] = mu.classifier_dual(swap_kernel_test_train, alpha_dual_swap, y_train, b_dual_swap,SV_dual_swap)    
        accuracy_dual_swap[shots_index,i] = mu.accuracy_score(y_test, y_pred_dual_swap[shots_index,i,:])

        y_pred_primal_gates[shots_index,i,:] = mu.classifier_primal(gates_kernel_test_train, beta_primal_gates, b_primal_gates)    
        accuracy_primal_gates[shots_index,i] = mu.accuracy_score(y_test, y_pred_primal_gates[shots_index,i,:])

        y_pred_dual_gates[shots_index,i,:] = mu.classifier_dual(gates_kernel_test_train, alpha_dual_gates, y_train, b_dual_gates, SV_dual_gates)    
        accuracy_dual_gates[shots_index,i] = mu.accuracy_score(y_test, y_pred_dual_gates[shots_index,i,:])

        y_pred_nsvm_gates[shots_index,i,:] = nominal_svm.predict(gates_kernel_test_train)    
        accuracy_nsvm_gates[shots_index,i] = mu.accuracy_score(y_test, y_pred_nsvm_gates[shots_index,i,:])#Nominal SVM. Uses GATES stochastic kernel matrix.

        y_pred_nsvm_swap[shots_index,i,:] = nominal_svm.predict(swap_kernel_test_train)    
        accuracy_nsvm_swap[shots_index,i] = mu.accuracy_score(y_test, y_pred_nsvm_swap[shots_index,i,:])#Nominal SVM. Uses SWAP stochastic kernel matrix.

    rely_primal_swap[shots_index] = mu.robustness(y_pred_primal_swap[shots_index,:,:], y_test, fraction = 1)
    rely_primal_gates[shots_index] = mu.robustness(y_pred_primal_gates[shots_index,:,:], y_test, fraction = 1)
    rely_dual_swap[shots_index] = mu.robustness(y_pred_dual_swap[shots_index,:,:], y_test, fraction = 1)
    rely_dual_gates[shots_index] = mu.robustness(y_pred_dual_gates[shots_index,:,:], y_test, fraction = 1)
    rely_nsvm_gates[shots_index] = mu.robustness(y_pred_nsvm_gates[shots_index,:,:],y_test, fraction = 1)
    rely_nsvm_swap[shots_index] = mu.robustness(y_pred_nsvm_swap[shots_index,:,:],y_test, fraction = 1)

    frac_rely_primal_swap[shots_index] = mu.robustness(y_pred_primal_swap[shots_index,:,:], y_test, fraction = 2/3)
    frac_rely_primal_gates[shots_index] = mu.robustness(y_pred_primal_gates[shots_index,:,:], y_test, fraction = 2/3)
    frac_rely_dual_swap[shots_index] = mu.robustness(y_pred_dual_swap[shots_index,:,:], y_test, fraction = 2/3)
    frac_rely_dual_gates[shots_index] = mu.robustness(y_pred_dual_gates[shots_index,:,:], y_test, fraction = 2/3)
    frac_rely_nsvm_gates[shots_index] = mu.robustness(y_pred_nsvm_gates[shots_index,:,:],y_test, fraction = 2/3)
    frac_rely_nsvm_swap[shots_index] = mu.robustness(y_pred_nsvm_swap[shots_index,:,:],y_test, fraction = 2/3)
accuracy_av_swap = np.mean(accuracy_nsvm_swap,axis=1)
accuracy_av_gates= np.mean(accuracy_nsvm_gates,axis=1)
plt.figure(4)
plt.plot(np.log(shots_array),accuracy_av_swap,'X-')
plt.plot(np.log(shots_array),accuracy_av_gates,'h-')
plt.legend('SWAP','GATES',loc='lower right')
plt.show()
print(accuracy_av_swap.shape)
#sys.exit()
plt.figure(1)#Comment out either SWAP or GATE
plt.plot(np.log(shots_array),rely_primal_swap,'X-')
plt.plot(np.log(shots_array),rely_dual_swap,'d-')
plt.plot(np.log(shots_array),rely_nsvm_swap,'*-')
plt.plot(np.log(shots_array),rely_primal_gates,'P-')
plt.plot(np.log(shots_array),rely_dual_gates,'D-')
plt.plot(np.log(shots_array),rely_nsvm_gates,'h-')
plt.legend(\
        ['Primal:SWAP','Dual: SWAP','NSVM:SWAP',\
        'Primal:GATES','Dual:GATES','NSVM:GATES'],\
        loc = "lower right",fontsize = 15)
#plt.plot(np.log(shots_array),np.ones(shots_index+1),'k--')
#plt.plot(np.log(shots_array),np.ones(shots_index+1),'k--')
#plt.plot(np.log(shots_array),0.5*np.ones(shots_index+1),'k--')
plt.title('Reliability |Havlicek Kernel| Checkerboard Dataset',fontsize=15)
#plt.plot(np.log(shots_array),np.ones(shots_index+1),'k--')
#plt.plot(np.log(shots_array),0*np.ones(shots_index+1),'k--')
plt.xlabel(r'$\log(N)$',fontsize=15)
plt.ylabel(r'$R(N,\eta = 1)$',fontsize=15)
plt.show()
plt.figure(2)
plt.plot(np.log(shots_array),frac_rely_primal_swap,'X-')
plt.plot(np.log(shots_array),frac_rely_dual_swap,'d-')
plt.plot(np.log(shots_array),frac_rely_nsvm_swap,'*-')
plt.plot(np.log(shots_array),frac_rely_primal_gates,'P-')
plt.plot(np.log(shots_array),frac_rely_dual_gates,'D-')
plt.plot(np.log(shots_array),frac_rely_nsvm_gates,'h-')
plt.legend(\
        ['Primal:SWAP','Dual:SWAP','NSVM:SWAP',\
        'Primal:GATES','Dual:GATES','NSVM:GATES'],\
        loc = "lower right",fontsize = 15)
#plt.plot(np.log(shots_array),np.ones(shots_index+1),'k--')
#plt.plot(np.log(shots_array),np.ones(shots_index+1),'k--')
#plt.plot(np.log(shots_array),0.5*np.ones(shots_index+1),'k--')
plt.title('Fractional (2/3) Reliability |Havlicek Kernel| Checkerboard Dataset',fontsize=15)
#plt.plot(np.log(shots_array),np.ones(shots_index+1),'k--')
#plt.plot(np.log(shots_array),0*np.ones(shots_index+1),'k--')
plt.xlabel(r'$\log(N)$',fontsize=15)
plt.ylabel(r'$R(N,\eta = 2/3)$',fontsize=15)
plt.show()
plt.figure(3)
plt.plot(np.log(shots_array),dual_opt_val_array[:,0],'X-')
plt.plot(np.log(shots_array),dual_opt_val_array[:,1],'d-')
plt.plot(np.log(shots_array),primal_opt_val_array[:,0],'P-')
plt.plot(np.log(shots_array),primal_opt_val_array[:,1],'D-')
plt.xlabel(r'$\log(N)$', fontsize=15)
plt.ylabel('Optimum Value', fontsize=15)
plt.legend(['DUAL:GATES','DUAL:SWAP','Primal:GATES','Primal:SWAP'],loc = "lower right",fontsize = 15)
plt.show()

