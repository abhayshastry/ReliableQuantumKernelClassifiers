from sgd_robust import *


##Test 1 :  To see if the SGD code agrees with SVC with out the corerctions
## Beta values will not agree. SO it it sbetter to check the predictions on test and train datasets
rerun = True
path = "data.pkl"



def run_svc_compare(args):
    print(f'[JOB {args}]', flush=True)

    key, m, m_test,  C, = args
    K, y = return_kernel(key, m + m_test )
    K_train, y_train = K[:,:m][:m,:],  y[:m]
    K_test_train = K[:,m:][:m, :]
    y_test = y[m:]
    if C == "optimal":
        C = find_optimal_C(K_train,y_train, C_range = [0.5,1,3,5,10])

    clf = SVC(kernel="precomputed",C=C).fit(K_train,y_train)
    y1_svc = clf.decision_function(K_train)
    y2_svc = clf.decision_function(K_test_train.T)
    beta_sgd, b_sgd = primal_rob_SGD(K,y, learning_rate = lambda t : 100/(m*t), T = 10000)
    y1_sgd = K_train@beta_sgd + b_sgd
    y2_sgd = K_test_train@beta_sgd + b_sgd
    return y1_svc, y2_svc, y1_sgd, y2_sgd


y1_svc, y2_svc, y1_sgd, y2_sgd= run_svc_compare( (("QAOA","Generated"), 20,20,1  ))
