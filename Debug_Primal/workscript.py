from workscript_utils import *
import utils.math_utils as mu
import time
X,y = get_dataset("make_moons")
n_qubits= X.shape[1]
node = create_device("havlicek",n_qubits=n_qubits)
K_star,K_star_test_train,y_train,y_test = evaluate_kernel(X,y,node)
K_gauss,K_gauss_test_train = evaluate_kernel(X,y,"gaussian")
print(mu.eigmin(K_star))
print(mu.eigmin(K_gauss))
start = time.time()
beta_1,b_1,SV1 = nominal_classifier(K_star,y_train,svm_type="primal")
beta_2,b_2,SV2 = nominal_classifier(K_star,y_train,svm_type="dual")
beta_3,b_3,SV3 = nominal_classifier(K_star,y_train,svm_type="SVC")


print("beta_1:",beta_1,"b_1",b_1)
print("beta_2:",beta_2,"b_2",b_2)
print("beta_3:",beta_3,"b_3",b_3)
