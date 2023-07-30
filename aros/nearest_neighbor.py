'''
@brief k-d tree and searching NN
'''

import numpy as np
from utils.dist import calc_dist
from sklearn.neighbors import KDTree

def nn_bf(point,k,X):
    '''
    @brief brute force way for searching the nearest neighbor
    '''
    # Calculate distances
    tmp_dist = []
    for data in X:
        tmp_dist.append(calc_dist(point,data))
    count = 0 
    # Get k neighbors  
    pts = [] 
    inds = []
    dists = []
    while(count < k): 
        index = np.argmin(tmp_dist) 
        pts.append(X[index])
        inds.append(index)
        dists.append(tmp_dist[index])
        # self.renew_neighbor(X[index],index,y[index])
        tmp_dist[index] = float('inf')
        count += 1
    return pts,inds,dists
    
def nn_kd(point,k,tree):
    '''
    @brief searching the nearest neighbor by created k-d tree
    '''
    dists, inds = tree.query(point, k)
    return inds[0],dists[0]
    
def create_kd(X):
    
    return KDTree(X, leaf_size=2)