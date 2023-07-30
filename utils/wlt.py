'''
@brief WLT test functions
@author yizhi
'''
import numpy as np
import scipy.stats as s


def WLT_test(a,b,conf=0.05):
    t_ = t(a,b)
    # print('t:',t_)
    v_ = v(a,b)
    # print('v:',v_)
    p_ = p(t_,v_)
    # print('p:',p_)
    if p_ > conf:
        return 'tie'
    else:
        if t_ > 0 :
            return 'win'
        else:
            return 'loss'
        
        
def var(vector):
    mean=sum(vector)/len(vector)
    sigma=0
    for i in vector:
        sigma+=(i-mean)**2
    variance_data=sigma / (len(vector)-1)
#     sd=math.sqrt(variance_data)
    return variance_data


def t(a,b):
    a,b = np.array(a),np.array(b)
    a_mean,b_mean = a.mean(),b.mean()
    a_var,b_var = var(a),var(b)
    n_a,n_b = len(a),len(b)
    return (a_mean-b_mean)/np.sqrt((a_var/n_a + b_var/n_b))


def v(a,b):
    a,b = np.array(a),np.array(b)
    a_var,b_var = var(a),var(b)
    n_a,n_b = len(a),len(b)
    return (a_var/n_a + b_var/n_b) ** 2 /  ((a_var/n_a)**2/(n_a-1) + (b_var/n_b)**2/(n_b-1))


def p(t,v):
    return 2 * (1 - s.t.cdf(abs(t), v))