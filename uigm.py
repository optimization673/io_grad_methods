import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime

Y0     = 0
STEP   = 0.15

def FastDualAveraging(target, x0, eps, set_type = "rn", grad = approx_fprime, step = lambda k: STEP, alpha = lambda k : 1/L):
    index = 0
    y_new = Y0
                      
    grad_sum = np.zeros((len(x0),))
    zk = [x0]
    xk = [x0]
    
    while True :
        index += 1
        y_old = y_new
        #x_k+1
        x_new = alpha(index)*y_old + (1 - alpha(index))*zk[index - 1]
        xk.append(x_new)
        grad_sum += grad(x_new, target, 0.0000001)*step(index)
        #y_k+1 (шаг метода двойственных усреднений зависит от целевого множества)
        if set_type == "rn" :
            y_new = (-1/2*grad_sum)
        elif set_type == "simplex" :
            y_new = (np.exp(-(np.ones(len(x0)) + grad_sum)))
        else :
            print("you oblazhalsya! (rn and simplex are available only)")
            return zk
        #z_k+1
        zk.append(alpha(index)*(y_old) + (1 - alpha(index))*zk[index - 1])
        #exit condition
        #print(norm(y_new - y_old, ord = 2))
        #print(norm(zk[index] - zk[index-1], ord = 2))
        #if norm(zk[index] - zk[index-1], ord = 2) < eps/2 :
        #    return zk
        if index >= 10000 :
            return zk
        


def Gradient(target, x0, eps, L, set_type = "rn", grad = approx_fprime):
    xk = [x0]
    index = 0
    while True:
        index += 1
        xk.append(xk[index - 1] - 1/L*grad(xk[index - 1], target, 0.0000001))
        #if norm(xk[index] - xk[index - 1], ord = 2) < eps / 2 :
        #    return xk
        if index >= 10000 :
            return xk
