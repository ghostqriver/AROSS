import numpy as np
from scipy.spatial import distance
from pyclustering.utils import euclidean_distance_square

L_distances= {'manhatan': 1, 'euclidean': 2, 'mahalanobis': 3}

def calc_cov_i(data):
    cov = np.cov(data,rowvar=False)
    cov_i = np.linalg.inv(cov)
    return cov_i
    
def mahalanobis(x,y,cov_i=None):
    if cov_i is None: 
        raise BaseException('cov_i is None')
    return distance.mahalanobis(x,y,cov_i)

def mahalanobis_distance_square(x,y,cov_i):
    # The fase mode for saving time in CURE
    if cov_i is None:
        raise BaseException('cov_i is None')
    delta = np.subtract(x,y)
    m = np.dot(np.dot(delta, cov_i), delta)
    return m
    
def euclidean(x,y):
    return distance.euclidean(x,y)

def manhantan(x,y):
    return np.abs(x - y).sum()

def calc_dist(x,y,L,cov_i=None):
    
    if L == 1:
        return manhantan(x,y)
    if L == 2:
        return euclidean(x,y)
    
    return mahalanobis(x,y,cov_i=cov_i)

def fast_dist(x,y,L,cov_i=None):
    # Fast mode
    if L == 1:
        return manhantan(x,y)
    if L == 2:
        return euclidean_distance_square(x,y)
    
    return mahalanobis_distance_square(x,y,cov_i=cov_i)
