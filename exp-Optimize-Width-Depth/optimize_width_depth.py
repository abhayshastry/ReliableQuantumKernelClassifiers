from exp_utils import *

K,y = return_kernel(("QAOA,2,1", "Checkerboard" ), 10)


def optimize_width_depth_train_err(kernel_key, dataset_key,C, m, width_range=[1,2,3], depth_range = [1,2,3,4,5,6,7] ):
    np.random.seed(0) ##For repeatability

    for w in width_range:
        for d  in depth_range:
            key = kernel_key + "," + str(w) + "," + str(d)
            print(key)
            K_train, y_train = return_kernel((key,dataset_key), m )
            clf =  SVC(kernel='precomputed', C = C)
            clf.fit(K_train, y_train)
            f_pred_exact =  np.sign(clf.decision_function(K_train))
            train_err = 0.5*np.mean(f_pred_exact - y_train)
            print(f"{key}-TrainError ={train_err} ")

def optimize_width_depth_margin_err(kernel_key, dataset_key,C, m, width_range=[1,2,3], depth_range = [1,2,3,4,5,6,7] ):
    np.random.seed(0) ##For repeatability

    for w in width_range:
        for d  in depth_range:
            key = kernel_key + "," + str(w) + "," + str(d)
            print(key)
            K_train, y_train = return_kernel((key,dataset_key), m )
            clf =  SVC(kernel='precomputed', C = C)
            clf.fit(K_train, y_train)
            f_pred_exact =  clf.decision_function(K_train)
            margin_err =  np.mean(y_train*f_pred_exact < 1.0 - 1e-13)
            print(f"{key}-MarginError = {margin_err} ")


def optimize_width_depth_cv(kernel_key, dataset_key,C, m, cv_fraction=0.2, N_trials = 10, width_range=[1,2,3], depth_range = [1,2,3,4,5,6,7] ):
    np.random.seed(0) ##For repeatability
    raise:
    
    m_test = np.ceil(cv_fraction*m)
    m_train = m - m_test

    for w in width_range:
        for d  in depth_range:
            err_list = []
            for _ in range(N_trials):
                key = kernel_key + "," + str(w) + "," + str(d)
                print(key)
                K_train, y_train = return_kernel((key,dataset_key), m )
                ind_list = np.random.shuffle(m)
                K_train = 
                clf =  SVC(kernel='precomputed', C = C)
                clf.fit(K_train, y_train)
                f_pred_exact =  np.sign(clf.decision_function(K_train))
                train_err = 0.5*np.mean(f_pred_exact - y_train)
                print(f"{key}-TrainError = {train_err} ")
