import pennylane as qml
from pennylane.templates import AngleEmbedding, QAOAEmbedding, IQPEmbedding
import numpy as np
#Quantum

def device_maker(qubits):
    dev_kernel = qml.device('default.qubit', wires = qubits)
    return dev_kernel

def projector(qubits):
    #qubits = len(dev_kernel.wires)
    projector = np.zeros((2**qubits,2**qubits))
    projector[0,0] = 1
    return projector

def qaoa_kernel(x,y,qubits,weights, n_repeats = 1):
#    if weights.all() == None:
#        print('Warning: Weights are a required argument for QAOA kernel')
#        print('Returning Angle Kernel Instead')
#        return angle_kernel(x,y,qubits)
    P = projector(qubits)
    QAOAEmbedding(x,weights, wires= range(qubits))
    qml.adjoint(QAOAEmbedding)(y,weights, wires = range(qubits))
    return qml.expval(qml.Hermitian(P,wires = range(qubits)))
    
def angle_kernel(x1, x2, qubits, weights = None, n_repeats = 1):
    #Quantum Kernel. Defined by Angle Pauli-X Embedding
#    qubits = len(dev_kernel.wires)
    P = projector(qubits)
    AngleEmbedding(x1, wires=range(qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(qubits))
    return qml.expval(qml.Hermitian(P,wires=range(qubits)))

def havlicek_kernel(x,y,qubits,weights = None, n_repeats = 1):
#    qubits = len(dev_kernel.wires)
    P = projector(qubits)
    IQPEmbedding(x, wires= range(qubits), n_repeats = n_repeats)
    qml.adjoint(IQPEmbedding)(y, wires = range(qubits), n_repeats = n_repeats)
    return qml.expval(qml.Hermitian(P,range(qubits)))

def device_wrapper(qubits, quantum_kernel):#Wraps device and kernel evaluation circuit into a qnode
    return qml.QNode(quantum_kernel, device_maker(qubits))
    
def quantum_kernel_matrix(X,quantum_node, weights = None,n_repeats = 1):
    #Computes only the exact kernel. K^(N) computed via numpy. math_utils.stochastic_kernel_matrix
    L = len(X)
    KernelMatrix = np.zeros((L,L))
    for l in range(L):
        for m in range(l):
            KernelMatrix[l,m] = quantum_node(X[l],X[m],len(quantum_node.device.wires),weights,n_repeats=n_repeats)
    return KernelMatrix + KernelMatrix.T + np.identity(L) #quantum kernel evaluations take time. This reduces L^2 computations to the upper triangle L(L-1)/2.

def quantum_test_train_kernel(X_test, X_train, quantum_node, weights = None, n_repeats = 1):#Output has to have dim (L_test,L_train)
    L_test = len(X_test)
    L_train = len(X_train)
    TestTrainKernel = np.zeros((L_test,L_train))
    for l_test in range(L_test):
        for l_train in range(L_train):
            TestTrainKernel[l_test,l_train]= quantum_node(X_test[l_test],X_train[l_train],len(quantum_node.device.wires),weights, n_repeats = n_repeats)
    return TestTrainKernel
#end of quantum
