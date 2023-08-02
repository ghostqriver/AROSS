'''
@author yizhi
'''
from aros.aros import AROS as AROS_

class AROSS():
    def __init__(self,n_cluster:int = None,linkage:str = None,alpha = 0,L = 2,IR = 1,all_safe_weight = 1):
        self.N = n_cluster
        self.linkage = linkage
        self.alpha = alpha
        self.L = L
        self.IR = IR
        self.all_safe_weight=all_safe_weight
        
    def fit_sample(self,X,y):
        return AROS_(X,y,N=self.N,linkage=self.linkage,alpha=self.alpha,L=self.L,IR=self.IR,all_safe_weight=self.all_safe_weight)
