from pennylane import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator


def parametric_quantum_kernel_1(n):

    params = [Parameter( "x" + str(i) ) for i in range(n)]

    qc =  QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        qc.rz(params[i], i)

    for i in range(n-1):
        if i != (i+1)%n:
            qc.cx(i, (i+1)%n)

    qc.barrier()

    for i in range(n):
        qc.rz(params[i], i)

    for i in range(n-1):
        if i != (i+1)%n:
            qc.cx(i, (i+1)%n)
    return qc,params

def data_gen(beta, b, margin, qc,  params,n,N,  balance = 0.5, scale = 1.0):
    dim = 2 ** n
    m_sv =  beta.shape[0]
    X_list = []
    y_list = np.zeros(N)
    supp_vecs = scale*(  np.random.rand(n, m_sv) - 0.5)
    supp_vecs_features =  np.zeros((dim, dim, m_sv), dtype = np.complex128)


    
    ket_0 =  np.zeros((2**n,1))
    ket_0[0,0] = 1.0
    for ind in range(m_sv):
        vals = {}
        for k in range(n):
            vals[params[k]] =  supp_vecs[k,ind]
        circ = qc.bind_parameters(vals)
        U =  Operator(circ).data
        Phi_x = U@ket_0
        supp_vecs_features[:, :,ind] = Phi_x* np.conj(Phi_x.T)

    w =  np.einsum("ijk,k->ij",supp_vecs_features, beta)

    linear_func =  lambda x : np.dot(x.flatten(), w.flatten()) + b  
    counter = 0

    while len(X_list) < N:
        rand_x =  scale*( np.random.rand(n) - 0.5)
        vals = {}
        for k in range(n):
            vals[params[k]] = rand_x[k] 
        circ = qc.bind_parameters(vals)
        U =  Operator(circ).data
        Phi_x = U@ket_0
        f_x =  linear_func(Phi_x*np.conj(Phi_x.T))
        if abs(f_x) >= margin : 
            y_list[counter] = np.sign(f_x)

            if np.abs( np.mean(y_list)) <  balance + 0.1:
                counter += 1
                X_list.append(rand_x[:])
        
        print (f"Data_generated:{len(X_list)*100/N}% , func_val = {f_x}, balance = { np.mean(y_list)}", end = "\r")
        X_sv = [supp_vecs[:,i] for i in range(m_sv)]
        y_sv =  np.sign(beta)
    return X_list, y_list, X_sv, y_sv
        
        
        
        


def kernel_data_gen(X,qc, params , M, bias = 0.0):
    (N,n) = X.shape
    y =  np.zeros(N)
     ##   \bra{0} U^\dagger(x) M  U(x)\ket{0}
    ket_0 =  np.zeros((2**n,1))
    ket_0[0,0] = 1.0
    ind_list = []
    for k in range(N):
        vals = {}
        for ind in range(n):
            vals[params[ind]] =  X[k,ind]
        circ = qc.bind_parameters(vals)
        U =  Operator(circ).data
        Phi_x = U@ket_0
        f1 = (np.conj(Phi_x.T)@M@Phi_x)[0,0] 
        
        if f1 >  bias:
            y[k] = 1
            ind_list.append(k)
        if f1 <  -bias:
            y[k] = -1
            ind_list.append(k)
    return y, ind_list

def q_features_from_circ(x, qc, params ):
    n =  len(params)
    ket_0 =  np.zeros((2**n,1))
    ket_0[0,0] = 1.0
    vals = {}
    for ind in range(n):
        vals[params[ind]] =  x[ind]
    circ = qc.bind_parameters(vals)
    U =  Operator(circ).data
    return U@ket_0

