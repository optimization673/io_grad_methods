import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime

Y0     = 0
STEP   = 0.15

def FastDualAveraging(target, x0, eps, set_type = "rn", grad = approx_fprime, step = lambda k: STEP, alpha = lambda k : 1/L, num_of_steps = 10000):
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
        if index >= num_of_steps :
            return zk
        


def Gradient(target, x0, eps, L, set_type = "rn", grad = approx_fprime, num_of_steps = 10000):
    xk = [x0]
    index = 0
    while True:
        index += 1
        xk.append(xk[index - 1] - 1/L*grad(xk[index - 1], target, 0.0000001))
        #if norm(xk[index] - xk[index - 1], ord = 2) < eps / 2 :
        #    return xk
        if index >= num_of_steps :
            return xk

def AdaptiveFDA(target, x0, eps, set_type = "rn", grad = approx_fprime, num_of_steps = 10000):
    index = 0
    y_new = Y0
    
    grad_sum = np.zeros((len(x0),))
    L_adapt = 1.
    alpha = 2./(index+1)
    zk = [x0]
    xk = [x0]
    
    while True :
        #print("index = ", index, ", L = ", L_adapt)
        index += 1
        y_old = y_new
        
        exit_condition = False
        while not exit_condition:
            
            alpha = 2./(index+1)
            step = (index+1.)/(2*L_adapt)
            #x_k+1
            x_new = alpha*y_old + (1 - alpha)*zk[index - 1]
        
            grad_sum_tmp = grad_sum + grad(x_new, target, 0.0000001)*step
            #y_k+1 (шаг метода двойственных усреднений зависит от целевого множества)
            if set_type == "rn" :
                y_new = (-1/2*grad_sum_tmp)
            elif set_type == "simplex" :
                y_new = (np.exp(-(np.ones(len(x0)) + grad_sum_tmp)))
            else :
                print("you oblazhalsya! (rn and simplex are available only)")
                return zk

            if target(x_new) > target(y_new) + (grad(y_new, target, 0.0000001) @ (x_new - y_new)) + L_adapt/2 * (norm(x_new - y_new, ord = 2) ** 2):
                L_adapt *= 2
                exit_condition = False
            else:
                xk.append(x_new)
                grad_sum = grad_sum_tmp
                #z_k+1
                zk.append(alpha*(y_old) + (1 - alpha)*zk[index - 1])
                exit_condition = True
        #exit condition
        #print(norm(y_new - y_old, ord = 2))
        #print(norm(zk[index] - zk[index-1], ord = 2))
        #if norm(zk[index] - zk[index-1], ord = 2) < eps/2 :
        #    return zk
        if index >= num_of_steps :
            return zk

def UniversalFDA(target, x0, eps, set_type = "rn", grad = approx_fprime, num_of_steps = 10000):
    index = 0
    y_new = Y0
    
    grad_sum = np.zeros((len(x0),))
    L_adapt = 1.
    alpha = 1 / L_adapt
    Ak = alpha 
    zk = [x0]
    xk = [x0]
    
    while True :
        #print("index = ", index, ", L = ", L_adapt)
        index += 1
        y_old = y_new
        exit_condition = False
        while not exit_condition:
            
            alpha = 2./(index+1)
            step = (index+1.)/(2*L_adapt)
            #x_k+1
            x_new = alpha*y_old + (1 - alpha)*zk[index - 1]
        
            grad_sum_tmp = grad_sum + grad(x_new, target, 0.0000001)*step
            #y_k+1 (шаг метода двойственных усреднений зависит от целевого множества)
            if set_type == "rn" :
                y_new = (-1/2*grad_sum_tmp)
            elif set_type == "simplex" :
                y_new = (np.exp(-(np.ones(len(x0)) + grad_sum_tmp)))
            else :
                print("you oblazhalsya! (rn and simplex are available only)")
                return zk
            
            if target(zk[-1]) > target(x_new) + (grad(x_new, target, 0.0000001) @ (zk[-1] - x_new)) + L_adapt/2 * (norm(zk[-1] - x_new, ord = 2) ** 2) + eps*alpha/(2*Ak):
                L_adapt *= 2
                exit_condition = False
            else:
                xk.append(x_new)
                grad_sum = grad_sum_tmp
                #z_k+1
                zk.append(alpha*(y_old) + (1 - alpha)*zk[index - 1])
                exit_condition = True
        #exit condition
        #print(norm(y_new - y_old, ord = 2))
        #print(norm(zk[index] - zk[index-1], ord = 2))
        #if norm(zk[index] - zk[index-1], ord = 2) < eps/2 :
        #    return zk
        if index >= num_of_steps :
            return zk

