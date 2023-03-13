import numpy as np
from scipy.spatial import distance

L_distances= {'manhatan': 1, 'euclidean': 2, 'mahalanobis': 3}

def calc_cov_i(data):
    cov = np.cov(data,rowvar=False)
    cov_i = np.linalg.inv(cov)
    return cov_i
    
def mahalanobis(x,y,data=None,cov_i=None):
    if cov_i is None:
        if data is None:
            raise BaseException('When the cov_i is None, the data can not be None')
        else:
            cov_i = calc_cov_i(data)
    return distance.mahalanobis(x,y,cov_i)

def euclidean(x,y):
    return distance.euclidean(x,y)

def manhantan(x,y):
    return np.abs(x - y).sum()

def calc_dist(x,y,L=2,data=None,cov_i=None):
    
    if L == 1:
        return manhantan(x,y)
    if L == 2:
        return euclidean(x,y)
    
    return mahalanobis(x,y,data=data,cov_i=cov_i)
