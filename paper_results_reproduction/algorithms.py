import numpy as np
import random
import time
import pickle
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import norm as norm_d
from scipy.stats import expon
from scipy.stats import weibull_min as weibull
from scipy.stats import burr12 as burr
from scipy.stats import randint
from scipy.stats import bernoulli
import scipy
from functions import *
from copy import deepcopy


def clipped_sgd_const_stepsize_toy_expect(filename, x_init, L, gamma, distribution, lambd, clip_activation_iter=0,
         N=1000, max_t=np.inf, save_info_period=100):
    n = len(x_init)
    x = np.array(x_init)
    
    distrib_type = distribution[0]
    sigma = distribution[1]
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([0.5*L*(norm(x)**2)])
    sq_distances = np.array([norm(x) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    
    samples_counter = 0
    samples_number = min(N*n, 1000*n)
    assert(distrib_type in ['normal', 'exp', 'weibull', 'burr'])
    
    if (distrib_type == 'normal'):
        samples = norm_d.rvs(loc=0, scale=sigma, size=samples_number)
    if (distrib_type == 'exp'):
        samples = expon.rvs(loc=-sigma, scale=sigma, size=samples_number)
    if (distrib_type == 'weibull'):
        c = distribution[2]
        scale = sigma*1.0/np.sqrt( scipy.special.gamma(1+2.0/c) - ((scipy.special.gamma(1+1.0/c))**2) )
        shift = -scale*scipy.special.gamma(1+1.0/c)
        samples = weibull.rvs(c=c, loc=shift, scale=scale, size=samples_number)
    if (distrib_type == 'burr'):
        c = distribution[2]
        d = distribution[3]
        unscaled_var = d*scipy.special.beta((c*d-2)*1.0/c, (c+2)*1.0/c) - (d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c) ** 2)
        scale = sigma*1.0/np.sqrt(unscaled_var)
        shift = -scale*d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c)
        samples = burr.rvs(c=c, d=d, loc=shift, scale=scale, size=samples_number)
    
    
    for it in range(N):
        if samples_counter == samples_number:
            samples_counter = 0
            if (distrib_type == 'normal'):
                samples = norm_d.rvs(loc=0, scale=sigma, size=samples_number)
            if (distrib_type == 'exp'):
                samples = expon.rvs(loc=-sigma, scale=sigma, size=samples_number)
            if (distrib_type == 'weibull'):
                c = distribution[2]
                scale = sigma*1.0/np.sqrt( scipy.special.gamma(1+2.0/c) - ((scipy.special.gamma(1+1.0/c))**2) )
                shift = -scale*scipy.special.gamma(1+1.0/c)
                samples = weibull.rvs(c=c, loc=shift, scale=scale, size=samples_number)
            if (distrib_type == 'burr'):
                c = distribution[2]
                d = distribution[3]
                unscaled_var = d*scipy.special.beta((c*d-2)*1.0/c, (c+2)*1.0/c) - (d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c) ** 2)
                scale = sigma*1.0/np.sqrt(unscaled_var)
                shift = -scale*d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c)
                samples = burr.rvs(c=c, d=d, loc=shift, scale=scale, size=samples_number)
        rand_vec = samples[samples_counter:(samples_counter+n)]
        samples_counter += n
        g = L*x + rand_vec
        norm_g = norm(g)
        if it >= clip_activation_iter:
            if norm_g > lambd:
                g = np.multiply(g, lambd*1.0/norm_g)
        x = x - gamma * g
        num_of_data_passes += 1.0/1000
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, 0.5*L*(norm(x)**2))
            sq_distances = np.append(sq_distances, norm(x) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val,  0.5*L*(norm(x)**2))
        sq_distances = np.append(sq_distances, norm(x) ** 2)
    
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_clipped_SGD_const_stepsize_toy_expect_gamma_"+str(gamma)+"_lambda_"+str(lambd)
              +"_num_of_iters_"+str(N)
              +"_distrib_"+distribution[0]+"_clip_activates_"+str(clip_activation_iter)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def sgd_const_stepsize_toy_expect(filename, x_init, L, gamma, distribution, 
         N=1000, max_t=np.inf, save_info_period=100):
    n = len(x_init)
    x = np.array(x_init)
    
    distrib_type = distribution[0]
    sigma = distribution[1]
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([0.5*L*(norm(x)**2)])
    sq_distances = np.array([norm(x) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    
    samples_counter = 0
    samples_number = min(N*n, 1000*n)
    assert(distrib_type in ['normal', 'exp', 'weibull', 'burr'])
    
    if (distrib_type == 'normal'):
        samples = norm_d.rvs(loc=0, scale=sigma, size=samples_number)
    if (distrib_type == 'exp'):
        samples = expon.rvs(loc=-sigma, scale=sigma, size=samples_number)
    if (distrib_type == 'weibull'):
        c = distribution[2]
        scale = sigma*1.0/np.sqrt( scipy.special.gamma(1+2.0/c) - ((scipy.special.gamma(1+1.0/c))**2) )
        shift = -scale*scipy.special.gamma(1+1.0/c)
        samples = weibull.rvs(c=c, loc=shift, scale=scale, size=samples_number)
    if (distrib_type == 'burr'):
        c = distribution[2]
        d = distribution[3]
        unscaled_var = d*scipy.special.beta((c*d-2)*1.0/c, (c+2)*1.0/c) - (d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c) ** 2)
        scale = sigma*1.0/np.sqrt(unscaled_var)
        shift = -scale*d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c)
        samples = burr.rvs(c=c, d=d, loc=shift, scale=scale, size=samples_number)
    
    
    for it in range(N):
        if samples_counter == samples_number:
            samples_counter = 0
            if (distrib_type == 'normal'):
                samples = norm_d.rvs(loc=0, scale=sigma, size=samples_number)
            if (distrib_type == 'exp'):
                samples = expon.rvs(loc=-sigma, scale=sigma, size=samples_number)
            if (distrib_type == 'weibull'):
                c = distribution[2]
                scale = sigma*1.0/np.sqrt( scipy.special.gamma(1+2.0/c) - ((scipy.special.gamma(1+1.0/c))**2) )
                shift = -scale*scipy.special.gamma(1+1.0/c)
                samples = weibull.rvs(c=c, loc=shift, scale=scale, size=samples_number)
            if (distrib_type == 'burr'):
                c = distribution[2]
                d = distribution[3]
                unscaled_var = d*scipy.special.beta((c*d-2)*1.0/c, (c+2)*1.0/c) - (d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c) ** 2)
                scale = sigma*1.0/np.sqrt(unscaled_var)
                shift = -scale*d*scipy.special.beta((c*d-1)*1.0/c, (c+1)*1.0/c)
                samples = burr.rvs(c=c, d=d, loc=shift, scale=scale, size=samples_number)
        rand_vec = samples[samples_counter:(samples_counter+n)]
        samples_counter += n
        g = L*x + rand_vec
        x = x - gamma * g
        num_of_data_passes += 1.0/1000
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, 0.5*L*(norm(x)**2))
            sq_distances = np.append(sq_distances, norm(x) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val,  0.5*L*(norm(x)**2))
        sq_distances = np.append(sq_distances, norm(x) ** 2)
    
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_SGD_const_stepsize_toy_expect_gamma_"+str(gamma)+"_num_of_iters_"+str(N)
              +"_distrib_"+distribution[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def sstm(filename, x_init, A, y, a, L, 
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    xk = np.array(x_init)
    yk = deepcopy(xk)
    zk = deepcopy(xk)
    
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(yk, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(yk - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
    Ak = 0.0
    Ak1 = 0.0
    for it in range(int(S*m/batch_size)):
        alpha = (it+2)*1.0/(2*a*L)
        Ak1 = Ak + alpha
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size
        xk = (Ak*yk + alpha*zk)*1.0/Ak1
        g = logreg_grad(xk, [A_for_batch[batch_ind], y[batch_ind], l2, sparse_stoch])
        zk = zk - alpha*g
        yk = (Ak*yk + alpha*zk)*1.0/Ak1
        Ak = deepcopy(Ak1)
        
        num_of_data_passes += batch_size/m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(yk, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(yk - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(yk, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(yk - ref_point) ** 2)
    
    res = {'last_iter':yk, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_SSTM_a_"+str(a)+"_L_"+str(L)
              +"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def clipped_sstm(filename, x_init, A, y, a, B, L, 
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    xk = np.array(x_init)
    yk = deepcopy(xk)
    zk = deepcopy(xk)
    
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(yk, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(yk - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
    Ak = 0.0
    Ak1 = 0.0
    for it in range(int(S*m/batch_size)):
        alpha = (it+2)*1.0/(2*a*L)
        Ak1 = Ak + alpha
        lambd = B / alpha
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size
        xk = (Ak*yk + alpha*zk)*1.0/Ak1
        g = logreg_grad(xk, [A_for_batch[batch_ind], y[batch_ind], l2, sparse_stoch])
        norm_g = norm(g)
        if norm_g > lambd:
            g *= lambd/norm_g
        zk = zk - alpha*g
        yk = (Ak*yk + alpha*zk)*1.0/Ak1
        Ak = deepcopy(Ak1)
        
        num_of_data_passes += batch_size/m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(yk, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(yk - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(yk, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(yk - ref_point) ** 2)
    
    res = {'last_iter':yk, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_clipped-SSTM_a_"+str(a)+"_B_"+str(B)+"_L_"+str(L)
              +"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res


def clipped_sgd_const_stepsize_decr_clip(filename, x_init, A, y, gamma, lambd_schedule, 
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
        
    lambd = lambd_schedule[0]
    decr_period = lambd_schedule[1]
    decr_coeff = lambd_schedule[2]
    number_of_decreases = 0
    
    indices_counter = 0
    
    for it in range(int(S*m/batch_size)):
        if num_of_data_passes >= decr_period*(number_of_decreases+1):
            lambd *= decr_coeff
            number_of_decreases += 1
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size
        g = logreg_grad(x, [A_for_batch[batch_ind], y[batch_ind], l2, sparse_stoch])
        norm_g = norm(g)
        if norm_g > lambd:
            g *= lambd/norm_g
        x = prox_R(x - gamma * g, l1 * gamma)
        num_of_data_passes += batch_size/m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_clipped-SGD_decr_clip_const_stepsize_gamma_"+str(gamma)+"_init_lambda_"+str(lambd_schedule[0])
              +"_decr_period_"+str(decr_period)+"_decr_coeff_"+str(decr_coeff)
              +"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res


def clipped_sgd_const_stepsize(filename, x_init, A, y, gamma, lambd, 
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
    for it in range(int(S*m/batch_size)):
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size
        g = logreg_grad(x, [A_for_batch[batch_ind], y[batch_ind], l2, sparse_stoch])
        norm_g = norm(g)
        if norm_g > lambd:
            g *= lambd/norm_g
        x = prox_R(x - gamma * g, l1 * gamma)
        num_of_data_passes += batch_size/m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_clipped-SGD_const_stepsize_gamma_"+str(gamma)+"_lambda_"+str(lambd)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res


def sgd_const_stepsize(filename, x_init, A, y, gamma, 
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
    for it in range(int(S*m/batch_size)):
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size
        g = logreg_grad(x, [A_for_batch[batch_ind], y[batch_ind], l2, sparse_stoch])
        x = prox_R(x - gamma * g, l1 * gamma)
        num_of_data_passes += batch_size/m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_SGD_const_stepsize_gamma_"+str(gamma)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res











