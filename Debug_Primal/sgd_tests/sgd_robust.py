import sys
sys.path.append("../")
from q_kernels import *
from exp_utils import *
from qml_utils import *

def kappa(delta):
    return np.sqrt(2*np.log(1/delta))

def primal_rob_subgradient(beta, b,  params ):
    K,y,k1,k2,C = params
    b_norm = np.linalg.norm(beta)
    f = K@beta + b
    Ind = (1 + k1*b_norm - np.multiply(y,f)) > 0
    if b_norm >0:
        Term1 = k1*sum(Ind)*(beta/ b_norm) - K@( np.multiply(y, Ind))
    else:
        Term1 =  - K@( np.multiply(y, Ind))
    Term2 = (K + k2*np.eye(K.shape[0])) @beta
    Term3 = - sum(np.multiply(y, Ind))
    return C*Term1 + Term2  , Term3 

def primal_rob_obj(beta,b , params):
    K,y,k1,k2,C = params
    b_norm = np.linalg.norm(beta)
    yf = np.multiply(K@beta + b, y)
    A = 1 + k1*b_norm - yf
    Ind = (A > 0)* (A)
    return C*sum(Ind) + 0.5* beta.T @ ( K + k2*np.eye(K.shape[0])) @ beta


def primal_rob_SGD(K_train, y_train, shots= False, C = 1,
                   delta_1 = 0.1, delta_2 = 0.1, circuit_type = "gates",
                    batch_size = None, learning_rate = 0.001, momentum = 0.1, T = 100, numerical_shift = 1e-4):

    """
    References:[1] https://arxiv.org/pdf/1803.08823.pdf , for implementing SGD
    [2]Pegasos: primal estimated sub-gradient solver for SVM, for discussion on sub-gradients for the hinge-loss

    """

    g = 1
    if circuit_type == 'swap':
        g = 2

    if not shots:
        k1 = 0.0
        k2 = 0.0
    else:
        k1= kappa(delta_1)*(g/np.sqrt(shots))
        k2= kappa(delta_2)*(g/np.sqrt(shots))

    K_train = K_train + numerical_shift*np.eye(K_train.shape[0])

    params = (K_train, y_train, k1, k2,C)

    if not callable(learning_rate):
        lr_sch = lambda x: learning_rate ##Define a learning rate scheduler
    else:
        lr_sch = learning_rate

    m = y_train.shape[0]
    beta = np.zeros(m)
    b = 0.0
    dbeta = np.zeros(m)
    db = 0.0
    progress_bar = tqdm(range(1, T+1))
    for t in progress_bar :

        lr = lr_sch(t)
        if not momentum:
            ##This is regular SGD
            gradbeta, gradb = primal_rob_subgradient(beta, b,  params)
            dbeta, db = lr*dbeta, lr*db
        else:
            ## This is NAG
            gradbeta, gradb = primal_rob_subgradient(beta + momentum*dbeta, b + momentum*db,  params )
            dbeta = momentum*dbeta + lr*gradbeta
            db = momentum*db + lr*gradb

        ##Update
        beta =  beta - dbeta
        b = b -db
        ##Projection step
        beta[beta > C ] = C
        beta[beta < -C ] = -C

        obj = primal_rob_obj(beta, b,params)
        progress_bar.set_description(f"t, Obj: {t} , {obj}")


    print(f"T, Final Obj: {T} , {obj}" )
    return beta, b

'''
np.random.seed(0)
m = 17
K,y = return_kernel(("QAOA,2", "Checkerboard"), m )
beta, b = primal_rob_SGD(K,y, learning_rate = lambda t : 100/(m*t), T = 10000)
clf = SVC(kernel="precomputed", C = 1, verbose = True).fit(K,y)
beta_clf = np.zeros(m)
beta_clf[clf.support_] = clf.dual_coef_.flatten()
print(beta_clf)
print(beta)
'''

