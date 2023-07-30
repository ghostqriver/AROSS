'''
@ brief checking the validity and determining the parameter for AROS
'''

from cluster.cpcc import cpcc
from cluster.bic import choose_N
from .optimize import *

def param(X,y,N,linkage,alpha,L,IR,all_safe_weight=1):
    
    if linkage is None:
        linkage = cpcc(X)
    assert linkage in ['ward','single','complete','average'], 'The given linkage should be one of ward, single, complete or average'
        
    if N is None:
        N = choose_N(X,y,linkage,L=2)
    print(type(N))
    assert isinstance(N,int), 'n_cluster should be an integer' 
    
    assert isinstance(alpha,int) or isinstance(alpha,float) or alpha < 0 or alpha > 1, 'The given alpha should be in the interval [0,1]'
    
    assert L in [1,2,3] , 'The L should be 1,2 or 3, which 1 denotes Manhattan distance, 2 denotes Euclidean distance, 3 denotes Mahalanobis distance'
    
    assert isinstance(IR,int) or isinstance(IR,float), 'IR should be a number'
    
    assert isinstance(all_safe_weight,int) or isinstance(all_safe_weight,float), 'Weight should be a number'
        
    return linkage,N