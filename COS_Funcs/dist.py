import numpy as np
from scipy.spatial import distance


def mahalanobis(x,y,data=None,cov_i=None):
    if cov_i is None:
        if data is None:
            raise BaseException('When the cov_i is None, the data can not be None')
        else:
            cov = np.cov(data,rowvar=False)
            cov_i = np.linalg.inv(cov)
    return distance.mahalanobis(x,y,cov_i)


def euclidean(x,y):
    return distance.euclidean(x,y)


def dist(x,y,data=None,cov_i=None,distance='mahalanobis'):
    
    if distance == 'euclidean':
        return euclidean(x,y)
    return mahalanobis(x,y,data=None,cov_i=None)
